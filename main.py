import json
import os.path
import pickle
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, log_loss
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, Adagrad, RMSprop
from tqdm import tqdm

from net import NerLSTM

FOLDER_DATA = 'data'


def set_seeds(random_state=1):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)


def prepare_data():
    df = pd.read_csv(f'{FOLDER_DATA}/ner_datasetreference.csv', encoding='latin1')
    df = df.fillna(method='ffill')
    df.columns = ['sentence', 'word', 'pos', 'tag']

    # составление словаря, который каждому слову ставит число(индекс) для дальнейшего получения эмбеддинга
    # добавляем служебный токен PAD для приведения приложений в одну длину
    all_words = df.word.unique()
    word_dict = {word: i + 1 for i, word in enumerate(all_words)}
    word_dict['PAD'] = 0

    with open(f'{FOLDER_DATA}/word_dict.txt', 'w') as file:
        json.dump(word_dict, file)

    # сбор предложений в лист и преобразования слов в индексы
    sentences = df.groupby('sentence').apply(lambda x: " ".join(x.word))

    with open(f'{FOLDER_DATA}/sentences.txt', 'w') as file:
        file.write("\n".join(sentences))

    sentences_words = df.groupby('sentence').apply(lambda x: list(x.word))
    senteces_length = [len(x) for x in sentences_words]

    sentence_ids = ([torch.tensor([word_dict[word] for word in sentence], dtype=torch.int64)
                     for sentence in sentences_words])

    # сбор тэгов в коллекцию, тэг - класс для каждого слова
    sentences_tags = df.groupby('sentence').apply(lambda x: list(x.tag)).values
    unique_tags = df.tag.unique()
    tag2ind = {tag: i for i, tag in enumerate(unique_tags)}
    with open(f'{FOLDER_DATA}/tag2ind.txt', 'w') as file:
        json.dump(tag2ind, file)

    tags_targets = [torch.tensor([tag2ind[tag] for tag in sentence], dtype=torch.int64)
                    for sentence in sentences_tags]

    with open(f'{FOLDER_DATA}/tag2ind_dict.txt', 'w') as file:
        file.write(str(tag2ind))

    print("Минимальная длина предложения:", min(senteces_length))
    print("Максимальная длина предложения:", max(senteces_length))

    # паддинг предложений 0 токеном, чтобы длины предложений были одинаковы для объединения в тензор
    sentence_ids_padded = pad_sequence(sentence_ids).transpose(1, 0)
    tags_targets = pad_sequence(tags_targets, padding_value=-1).transpose(1, 0)

    print(sentence_ids_padded.size())

    pickle.dump(sentence_ids_padded, open(f'{FOLDER_DATA}/sentence_padded_data.pickle', 'wb'))
    pickle.dump(tags_targets, open(f'{FOLDER_DATA}/sentence_padded_target.pickle', 'wb'))

    return sentences, word_dict, sentence_ids_padded, tag2ind, tags_targets


def get_data():
    if os.path.exists(f'{FOLDER_DATA}/sentence_padded_target.pickle'):
        sentences = open(f'{FOLDER_DATA}/sentences.txt', 'r').read().split('\n')
        word_dict = json.load(open(f'{FOLDER_DATA}/word_dict.txt', 'r'))
        tag2ind = json.load(open(f'{FOLDER_DATA}/tag2ind.txt', 'r'))

        sentence_ids_padded = pickle.load(open(f'{FOLDER_DATA}/sentence_padded_data.pickle', 'rb'))
        tags_targets = pickle.load(open(f'{FOLDER_DATA}/sentence_padded_target.pickle', 'rb'))
    else:
        sentences, word_dict, sentence_ids_padded, tag2ind, tags_targets = prepare_data()

    return sentences, word_dict, sentence_ids_padded, tag2ind, tags_targets


def get_metrics_on_set(model: nn.Module, samples: torch.Tensor, targets: torch.Tensor, batch_size=50):
    predicts_classes_all = []
    predicts_probs_all = []
    targets_all = []

    for i in tqdm(range(0, samples.size()[0] - batch_size, batch_size)):
        res = model(samples[i:i + batch_size])
        res_classes = res.argmax(axis=-1)

        for j in range(batch_size):
            inds = targets[i + j] != -1

            predicts_probs_all.extend(res[j, inds].detach().numpy())
            predicts_classes_all.extend(res_classes[j, inds])
            targets_all.extend(targets[i + j, inds].detach().numpy())

    return pd.DataFrame(classification_report(targets_all, predicts_classes_all, output_dict=True)).T, \
           log_loss(y_true=targets_all, y_pred=predicts_probs_all)


def get_losses_epoch(sentence_ids_padded, tags_targets, weights):
    model = NerLSTM(len(word_dict) + 1, 64, 64, len(tag2ind))

    optimizer = Adam(model.parameters())
    loss_func = CrossEntropyLoss(ignore_index=-1, weight=weights, reduction='mean')

    batch_size = 40
    epoch_number = 15

    test_samples, test_targets = sentence_ids_padded[test_inds], tags_targets[test_inds]
    results = []

    for j in range(epoch_number):
        for i in tqdm(range(0, sentence_ids_padded[train_inds].shape[0] - batch_size, batch_size)):
            train_samples, train_targets = sentence_ids_padded[train_inds][i:i + 10], tags_targets[train_inds][i:i + 10]

            res = model(train_samples)

            loss = loss_func(res.view(-1, len(tag2ind)), train_targets.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        res_df, loss_value = get_metrics_on_set(model, test_samples, test_targets)
        results.append((j, loss_value, res_df.loc['accuracy'].iloc[0]))

    pd.DataFrame(results, columns=['epoch number', 'log loss', 'accuracy']) \
        .to_excel('epoch_loss_data.xlsx', index=False)


def get_losses_lstm_hidden_dim(sentence_ids_padded, tags_targets, weights):
    batch_size = 40
    epoch_number = 10

    test_samples, test_targets = sentence_ids_padded[test_inds], tags_targets[test_inds]
    results = []

    hidden_dims = [16, 32, 64, 100, 128]

    for dim in hidden_dims:
        set_seeds(47)
        print(dim)
        model = NerLSTM(len(word_dict) + 1, 64, dim, len(tag2ind))
        optimizer = Adam(model.parameters())

        loss_func = CrossEntropyLoss(ignore_index=-1, weight=weights, reduction='mean')

        for j in range(epoch_number):
            for i in tqdm(range(0, sentence_ids_padded[train_inds].shape[0] - batch_size, batch_size)):
                train_samples, train_targets = sentence_ids_padded[train_inds][i:i + 10], tags_targets[train_inds][
                                                                                          i:i + 10]

                res = model(train_samples)

                loss = loss_func(res.view(-1, len(tag2ind)), train_targets.view(-1))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        res_df, loss_value = get_metrics_on_set(model, test_samples, test_targets)
        print(res_df)
        results.append((dim, loss_value, res_df.loc['accuracy'].iloc[0]))

    pd.DataFrame(results, columns=['neurons number', 'log loss', 'accuracy']).to_excel('hidden_dim_loss_data.xlsx',
                                                                                       index=False)


def get_losses_lstm_learning_rate(sentence_ids_padded, tags_targets, weights):
    batch_size = 40
    epoch_number = 10

    test_samples, test_targets = sentence_ids_padded[test_inds], tags_targets[test_inds]
    results = []

    learning_rates = [0.0001, 0.001, 0.01, 0.05, 0.1]

    for lr in learning_rates:
        set_seeds(47)
        print(lr)
        model = NerLSTM(len(word_dict) + 1, 64, 64, len(tag2ind))
        optimizer = Adam(model.parameters(), lr=lr)

        loss_func = CrossEntropyLoss(ignore_index=-1, weight=weights, reduction='mean')

        for j in range(epoch_number):
            for i in tqdm(range(0, sentence_ids_padded[train_inds].shape[0] - batch_size, batch_size)):
                train_samples, train_targets = sentence_ids_padded[train_inds][i:i + 10], tags_targets[train_inds][
                                                                                          i:i + 10]

                res = model(train_samples)

                loss = loss_func(res.view(-1, len(tag2ind)), train_targets.view(-1))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        res_df, loss_value = get_metrics_on_set(model, test_samples, test_targets)
        print(res_df)
        results.append((lr, loss_value, res_df.loc['accuracy'].iloc[0]))

    pd.DataFrame(results, columns=['learning rate', 'log loss', 'accuracy']).to_excel('learning_rate_loss_data.xlsx',
                                                                                      index=False)


if __name__ == '__main__':
    set_seeds(47)

    sentences, word_dict, sentence_ids_padded, tag2ind, tags_targets = get_data()

    if os.path.exists('test_inds.pickle'):
        train_inds = pickle.load(open('train_inds.pickle', 'rb'))
        test_inds = pickle.load(open('test_inds.pickle', 'rb'))
    else:
        print(tags_targets.size())
        # # получениe номеров предложений для обучающей и тестовой выборки
        inds = np.arange(0, len(sentences), 1).tolist()
        num_train, num_test = int(len(sentences) * 0.8), len(sentences) - int(len(sentences) * 0.8)

        train_inds = random.sample(inds, k=num_train)
        test_inds = [ind for ind in inds if ind not in train_inds]

        pickle.dump(train_inds, open('train_inds.pickle', 'wb'))
        pickle.dump(test_inds, open('test_inds.pickle', 'wb'))

    print(len(train_inds), len(test_inds))

    model = NerLSTM(len(word_dict) + 1, 64, 64, len(tag2ind))

    target_count = pd.Series(tags_targets.flatten()).value_counts()
    target_count.drop(index=-1, inplace=True)
    target_count.sort_index(inplace=True)

    weights = torch.tensor(target_count.values) / target_count.sum()
    weights = 1. / weights

    # get_losses_epoch(sentence_ids_padded, tags_targets, weights)
    # get_losses_lstm_hidden_dim(sentence_ids_padded, tags_targets, weights)
    # get_losses_lstm_learning_rate(sentence_ids_padded, tags_targets, weights)

    loss_func = CrossEntropyLoss(ignore_index=-1, weight=weights, reduction='mean')

    BATCH_SIZE = 80
    EPOCH_NUMBER = 35

    test_samples, test_targets = sentence_ids_padded[test_inds], tags_targets[test_inds]

    optimizer = Adam(model.parameters())

    optimizers_dict = {'Adam': Adam, 'Adagrad': Adagrad, 'RMSprop': RMSprop}
    results = []

    for optimizer_name in optimizers_dict:
        optimizer_const = optimizers_dict[optimizer_name]

        results.append([])
        results[-1].append(optimizer_name)

        for lm in [0, 0.001]:
            print(lm)
            set_seeds(47)

            model = NerLSTM(len(word_dict) + 1, 64, 64, len(tag2ind))
            optimizer = optimizer_const(model.parameters())

            for j in range(EPOCH_NUMBER):
                print('Epoch:', j)
                for i in tqdm(range(0, sentence_ids_padded[train_inds].shape[0] - BATCH_SIZE, BATCH_SIZE)):
                    train_samples, train_targets = sentence_ids_padded[train_inds][i:i + 10], tags_targets[train_inds][i:i + 10]

                    res = model(train_samples)

                    weights, bias = model.last_layer.parameters()

                    loss = loss_func(res.view(-1, len(tag2ind)), train_targets.view(-1)) + lm * ((weights ** 2).sum() +
                                                                                                 (bias ** 2).sum())

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            res_df, loss = get_metrics_on_set(model, sentence_ids_padded[train_inds], tags_targets[train_inds])
            results[-1].append(res_df.loc['accuracy'].iloc[0])
            results[-1].append(res_df.loc['macro avg'].iloc[0])
            results[-1].append(loss)

            res_df, loss = get_metrics_on_set(model, test_samples, test_targets)
            results[-1].append(res_df.loc['accuracy'].iloc[0])
            results[-1].append(res_df.loc['macro avg'].iloc[0])
            results[-1].append(loss)

    print(results)
    df_result = pd.DataFrame(results, columns=['optimizer', 'accuracy_without_train',
                                   'macro_f1_without_train', 'loss_without_train',
                                   'accuracy_without_test', 'macro_f1_without_test', 'loss_without_test',
                                   'accuracy_with_train',
                                   'macro_f1_with_train', 'loss_with_train',
                                   'accuracy_with_test', 'macro_f1_with_test', 'loss_with_test'])
    print(df_result)
    df_result.to_excel('optimizer_data.xlsx', index=False)
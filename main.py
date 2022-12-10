import os.path

import pandas as pd
import numpy as np
import torch
import random
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pad_sequence
import pickle
import json
from torch import nn

from net import NerLSTM


FOLDER_DATA = 'data'


def set_seeds(random_state=1):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)

def prepare_dataset():
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
        sentences, word_dict, sentence_ids_padded, tag2ind, tags_targets = prepare_dataset()

    return sentences, word_dict, sentence_ids_padded, tag2ind, tags_targets


if __name__ == '__main__':
    set_seeds(47)

    sentences, word_dict, sentence_ids_padded, tag2ind, tags_targets = get_data()

    # print(tags_targets.size())
    # # # получениe номеров предложений для обучающей и тестовой выборки
    # inds = np.arange(0, len(sentences), 1).tolist()
    # num_train, num_test = int(len(sentences) * 0.8), len(sentences) - int(len(sentences) * 0.8)
    #
    # train_inds = random.sample(inds, k=num_train)
    # test_inds = [ind for ind in inds if ind not in train_inds]

    model = NerLSTM(len(word_dict) + 1, 64, 64, len(tag2ind))

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)


    model.apply(init_weights)

    take_samples = 40
    targets_test = tags_targets[:take_samples].contiguous()

    print(pd.Series(targets_test[0]).value_counts())
    optimizer = Adam(model.parameters())
    loss_func = CrossEntropyLoss(ignore_index=-1)

    tokens_take = 13

    for i in range(500):
        res = model(sentence_ids_padded[:take_samples, :tokens_take].contiguous())
        a = res[0, 6]
        # print(res.size())
        # print(targets_test[:, :tokens_take].size())
        loss = loss_func(res.view(-1, len(tag2ind)), targets_test[:, :tokens_take].contiguous().view(-1))
        print(loss)

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=0.1,
            norm_type=2,
        )

        optimizer.step()
        optimizer.zero_grad()

    print(res.size())
    print(res.argmax(axis=-1))
    print(targets_test[:, :tokens_take])



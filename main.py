import pandas as pd
import numpy as np
import torch
import random
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import pickle

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
        file.write(str(word_dict))

    # сбор предложений в лист и преобразования слов в индексы
    sentences = df.groupby('sentence').apply(lambda x: " ".join(x.word))
    sentences_words = df.groupby('sentence').apply(lambda x: list(x.word))
    senteces_length = [len(x) for x in sentences_words]

    sentence_ids = ([torch.tensor([word_dict[word] for word in sentence], dtype=torch.int64)
                     for sentence in sentences_words])

    # сбор тэгов в коллекцию, тэг - класс для каждого слова
    tags = df.groupby('sentence').apply(lambda x: list(x.tag)).values
    unique_tags = df.tag.unique()
    tag2ind = {tag: i for i, tag in enumerate(unique_tags)}

    with open(f'{FOLDER_DATA}/tag2ind_dict.txt', 'w') as file:
        file.write(str(tag2ind))


    print("Минимальная длина предложения:", min(senteces_length))
    print("Максимальная длина предложения:", max(senteces_length))

    # паддинг предложений 0 токеном, чтобы длины предложений были одинаковы для объединения в тензор
    sentence_ids_padded = pad_sequence(sentence_ids)


    return sentences_words, word_dict, sentence_ids_padded, tags, tag2ind


if __name__ == '__main__':
    set_seeds(47)

    prepare_dataset()

    # # получения номеров предложений для обучающей и тестовой выборки
    # inds = np.arange(0, len(sentences), 1).tolist()
    # num_train, num_test = int(len(sentences) * 0.8), len(sentences) - int(len(sentences) * 0.8)
    #
    # train_inds = random.sample(inds, k=num_train)
    # test_inds = [ind for ind in inds if ind not in train_inds]



    # # получения эмбеддингов
    # print(sentence_ids_padded[:, test_inds].size())
    # layer = nn.Embedding(len(word_dict), 128, padding_idx=0)
    # res = layer(sentence_ids_padded[:, test_inds])
    #
    # print(res)

    #
    # print(df[df['tag'] == 'B-nat']['word'].unique())
    # print(df[(df['tag'] == 'B-nat') & (df['word'] == 'Bird')].iloc[0])
    #
    # print(df[df['sentence #'] == 'Sentence: 8882'])
    # print(df[(df[''] == 'B-nat') & (df['word'] == 'Bird')].iloc[0])
    #
    # print(df['tag'].unique())



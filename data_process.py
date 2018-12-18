import pandas as pd
import numpy as np
import tensorflow as tf
import os

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from configs import DEFINES

def load_data():

    chatbot_data = load_chatbot_data(DEFINES.chatbot_data_path)
    chatbot_tuple = make_tuple(chatbot_data)
    x, y = make_data(chatbot_tuple)

    inputs, labels, t2i, i2t, max_len, embedding_matrix = make_input_label(x, y)

    return inputs, labels, t2i, i2t, max_len, embedding_matrix

def make_input_label(x,y):

    max_len = int(round(np.percentile(np.array([len(tok) for tok in x]), 99)))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    inputs = tokenizer.texts_to_sequences(x)
    inputs = pad_sequences(inputs, maxlen=max_len, padding = 'post')
    labels = pad_sequences(y, maxlen=max_len, padding = 'post')
    t2i, i2t = make_vocab(tokenizer.word_index)

    embedding_matrix = get_embedding_matrix(x, i2t)

    return inputs, labels, t2i, i2t, max_len, embedding_matrix

def make_vocab(vocab):

    pad = 0
    t2i = {'PAD': pad}
    i2t = {pad: 'PAD'}

    for word, idx in vocab.items():
        t2i[word] = idx
        i2t[idx] = word

    return t2i, i2t


def make_embedding(x, i2t):

    from gensim.models import Word2Vec

    model = Word2Vec(x,
                size = DEFINES.embedding_dim,
                window=7,
                min_count=3,
                workers=4,
                sg=1,
                iter = 10,
                sample = 1e-3)

    embedding_matrix = special_token_embedding(model)
    unk = embedding_matrix[1]
    for i, t in i2t.items():
        if i<2:
            continue
        if t in model.wv.vocab:
            embedding_matrix.append(model.wv.word_vec(t))
        else:
            embedding_matrix.append(unk)

    np.save(open(DEFINES.embedding_matrix_path, 'wb'), np.stack(embedding_matrix))

def special_token_embedding(model):
    pad = np.zeros(shape = (DEFINES.embedding_dim), dtype=np.float32)
    unk = np.random.uniform(low=model.wv.vectors.min(), high=model.wv.vectors.max(), size=(DEFINES.embedding_dim))

    return [pad, unk]


def load_chatbot_data(file_path):

    chatbot_data = pd.read_csv(file_path)
    chatbot_all = pd.Series(chatbot_data['Q'].tolist() + chatbot_data['A'].tolist())

    return chatbot_all

def make_tuple(data):

    output = []
    for sentence in data:
        temp=[]
        for i,v in enumerate(sentence[:-1]):
            if v == ' ':
                continue
            if sentence[i+1] == ' ':
                temp.append((v, 1))
            else:
                temp.append((v, 0))
        temp.append((sentence[-1],0))
        output.append(temp)
    return output

def make_data(tuple_data):

    x = []
    y = []

    for data in tuple_data:
        x_temp = [tok[0] for tok in data]
        y_temp = [tok[1] for tok in data]
        x.append(x_temp)
        y.append(y_temp)

    return x, y

def get_embedding_matrix(x, i2t):

    if not os.path.isfile(DEFINES.embedding_matrix_path):
        make_embedding(x, i2t)

    return np.load(open(DEFINES.embedding_matrix_path, 'rb')).astype(np.float32)

def mapping_fn(x,y):
    features = {'inputs' : x}
    return features, y

def train_input_fn(inputs, labels):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.shuffle(len(inputs))
    dataset = dataset.batch(DEFINES.batch_size)
    dataset = dataset.map(mapping_fn)
    dataset = dataset.repeat(DEFINES.epoch)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

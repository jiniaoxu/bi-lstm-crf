import os

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from bi_lstm_crf_model import *

CHUNK_TAGS = ['B', 'I', 'S']


def _parse_data(fh, splits):
    text = fh.readlines()
    sent, t1, t2, chunk = [], [], [], []
    for i in text:
        if i != '\n':
            char, tag = i.split()
            t1.append(char)
            t2.append(tag)
            if char in splits:
                sent.append(t1)
                chunk.append(t2)
                t1, t2 = [], []
        elif len(t1) != 0:
            sent.append(t1)
            chunk.append(t2)
            t1, t2 = [], []
    fh.close()
    sent = pd.Series(sent)
    chunk = pd.Series(chunk)
    return sent, chunk


def process_data(corops_path, max_num_words=20000, max_sequence_len=150, splits=None):
    if splits is None:
        splits = ['。']
    sent, chunk = _parse_data(open(corops_path, 'r', encoding='UTF-8'), splits)
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(sent)
    sequences = tokenizer.texts_to_sequences(sent)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=max_sequence_len)

    tokenizer_01 = Tokenizer(num_words=len(CHUNK_TAGS) + 1)
    tokenizer_01.fit_on_texts(chunk)
    chunk = tokenizer_01.texts_to_sequences(chunk)

    chunk_index = tokenizer_01.word_index
    chunk = pad_sequences(chunk, maxlen=max_sequence_len)

    chunk = to_categorical(chunk)

    return data, chunk, word_index, chunk_index


def get_embedding_index(embedding_file):
    embedding_index = {}
    with open(os.path.join(embedding_file), encoding='UTF-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embedding_index[word] = coefs
    return embedding_index


def create_embedding_matrix(embeddings_index, word_index, model_config):
    embedding_matrix = np.zeros((model_config.max_num_words, model_config.embed_dim))
    for word, i in word_index.items():
        if i >= model_config.max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def sentence_to_vec(sentences: iter, word_index, model_config):
    x = list(map(lambda sentence: [word_index.get(w, 1) for w in sentence], sentences))
    x = pad_sequences(x, maxlen=model_config.max_sequence_len)
    return x

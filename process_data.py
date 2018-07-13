import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
from bi_lstm_crf_model import BiLSTMCRFModelConfigure

CHUNK_TAGS = ['B', 'I', 'S']


def _parse_data(fh):
    text = fh.readlines()
    sent, t1, t2, chunk = [], [], [], []
    for i in text:
        if i != '\n':
            t1.append(i.split()[0].lower())
            t2.append(i.split()[1])
        else:
            sent.append(t1)
            chunk.append(t2)
            t1, t2 = [], []
    fh.close()
    sent = pd.Series(sent)
    chunk = pd.Series(chunk)
    return sent, chunk


def process_data(corops_path, max_num_words=20000, max_sequence_len=100):
    sent, chunk = _parse_data(open(corops_path, 'r', encoding='UTF-8'))
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


def sentence_to_vec(sentence, word_index, model_config: BiLSTMCRFModelConfigure):
    x = [word_index.get(w, 1) for w in sentence]
    x = pad_sequences([x], maxlen=model_config.max_sequence_len)
    return x

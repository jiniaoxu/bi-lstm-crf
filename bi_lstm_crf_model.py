from process_data import *
from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout, TimeDistributed
from keras.models import Model
from keras_contrib.layers import CRF

import keras
import pickle

"""
使用keras实现的中文分词，原理基于论文：https://arxiv.org/abs/1508.01991
实际上是一个序列标注模型
"""


class BiLSTMCRFModelConfigure:

    def __init__(self, vocab_size: int
                 , chunk_size: int
                 , embed_dim=300
                 , bi_lstm_units=200
                 , max_sequence_len=150
                 , max_num_words=20000):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bi_lstm_units = bi_lstm_units
        self.max_sequence_len = max_sequence_len
        self.max_num_words = min(max_num_words, vocab_size)
        self.chunk_size = chunk_size

    def build_model(self, embeddings_matrix=None):
        num_words = min(self.max_num_words, self.vocab_size)
        word_input = Input(shape=(self.max_sequence_len,), dtype='int32', name="word_input")

        if embeddings_matrix is not None:
            word_embedding = Embedding(num_words, self.embed_dim,
                                       input_length=self.max_sequence_len,
                                       weights=[embeddings_matrix],
                                       trainable=False,
                                       name='word_emb')(word_input)
        else:
            word_embedding = Embedding(num_words, self.embed_dim, input_length=self.max_sequence_len, name="word_emb") \
                (word_input)
        bilstm = Bidirectional(LSTM(self.bi_lstm_units // 2, return_sequences=True))(word_embedding)
        x = Dropout(0.2)(bilstm)
        dense = TimeDistributed(Dense(self.chunk_size))(x)
        crf = CRF(self.chunk_size, sparse_target=False)
        crf_output = crf(dense)

        model = Model([word_input], [crf_output])

        model.compile(optimizer=keras.optimizers.Adam(), loss=crf.loss_function, metrics=[crf.accuracy])
        return model


def save_dict(dict: tuple, dict_path):
    with open(dict_path, "wb") as f:
        pickle.dump(dict, f)


def save_model_config(model_config: BiLSTMCRFModelConfigure
                      , model_config_path):
    with open(model_config_path, "wb") as f:
        pickle.dump(model_config, f)


def load_model_config(model_config_path) -> BiLSTMCRFModelConfigure:
    with open(model_config_path, 'rb') as f:
        model_builder = pickle.load(f)
    return model_builder


def load_dict(dict_path):
    with open(dict_path, 'rb') as f:
        vocab, chunk = pickle.load(f)
    return vocab, chunk

from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout, TimeDistributed
from keras.models import Model
from keras_contrib.layers import CRF
import pickle

"""
使用keras实现的中文分词，原理基于论文：https://arxiv.org/abs/1508.01991
实际上是一个序列标注模型
"""


class BiLSTMCRFModelConfigure:

    def __init__(self, word_index: dict
                 , chunk_index: dict
                 , embed_dim=200
                 , bi_lstm_units=200
                 , max_sequence_len=1000
                 , max_num_words=20000):
        self.word_index = word_index
        self.embed_dim = embed_dim
        self.bi_lstm_units = bi_lstm_units
        self.max_sequence_len = max_sequence_len
        self.max_num_words = max_num_words
        self.chunk_index = chunk_index

    def build_model(self):
        num_words = min(self.max_num_words, len(self.word_index) + 1)
        sequence_input = Input(shape=(self.max_sequence_len,))
        embedded_sequences = Embedding(num_words, self.embed_dim, input_length=self.max_sequence_len)(sequence_input)
        x = Bidirectional(LSTM(self.bi_lstm_units // 2, return_sequences=True))(embedded_sequences)
        x = Dropout(0.2)(x)
        x = TimeDistributed(Dense(len(self.chunk_index) + 1))(x)
        crf = CRF(len(self.chunk_index) + 1, sparse_target=True)
        x = crf(x)

        model = Model([sequence_input], x)

        model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
        return model


def save_model(model_config: BiLSTMCRFModelConfigure, model_config_path, model: Model, model_path):
    with open(model_config_path, "wb") as f:
        pickle.dump(model_config, f)
    model.save(model_path)


def load_model(weights_path, model_config_path):
    with open(model_config_path, 'rb') as f:
        model_builder: BiLSTMCRFModelConfigure = pickle.load(f)
    model = model_builder.build_model()
    model.load_weights(weights_path)
    return model, model_builder.word_index, model_builder.chunk_index

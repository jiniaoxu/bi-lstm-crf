from keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout, TimeDistributed
from keras.models import Model
from keras_contrib.layers import CRF
import pickle

"""
使用keras实现的中文分词，原理基于论文：https://arxiv.org/abs/1508.01991
实际上是一个序列标注模型
"""


class BiLSTMCRFModelConfigure:

    def __init__(self, vocab_size: int
                 , chunk_size: int
                 , embed_dim=200
                 , bi_lstm_units=200
                 , max_sequence_len=100
                 , max_num_words=20000):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bi_lstm_units = bi_lstm_units
        self.max_sequence_len = max_sequence_len
        self.max_num_words = max_num_words
        self.chunk_size = chunk_size

    def build_model(self):
        num_words = min(self.max_num_words, self.vocab_size)
        word_input = Input(shape=(self.max_sequence_len,), dtype='int32', name="word_input")
        word_emb = Embedding(num_words, self.embed_dim, input_length=self.max_sequence_len, name="word_emb")(word_input)
        bilstm = Bidirectional(LSTM(self.bi_lstm_units // 2, return_sequences=True))(word_emb)
        x = Dropout(0.2)(bilstm)
        dense = TimeDistributed(Dense(self.chunk_size))(x)
        crf = CRF(self.chunk_size, sparse_target=False)
        crf_output = crf(dense)

        model = Model([word_input], [crf_output])

        model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])
        return model


def save_model(model_config: BiLSTMCRFModelConfigure
               , model_config_path
               , model: Model
               , model_path
               , dict: tuple
               , dict_path):
    with open(model_config_path, "wb") as f:
        pickle.dump(model_config, f)
    with open(dict_path, "wb") as f:
        pickle.dump(dict, f)
    model.save(model_path)


def load_model(weights_path, model_config_path, dict_path):
    with open(model_config_path, 'rb') as f:
        model_builder: BiLSTMCRFModelConfigure = pickle.load(f)
    with open(dict_path, 'rb') as f:
        vocab, chunk = pickle.load(f)
    model = model_builder.build_model()
    model.load_weights(weights_path)
    return model, model_builder, vocab, chunk

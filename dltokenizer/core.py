import json

from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dropout, Dense, LSTM, Lambda
from keras.optimizers import Adam
from keras_contrib.layers import CRF
from dltokenizer.tools import load_dictionary


class DLTokenizer:
    __singleton = None

    def __init__(self,
                 vocab_size,
                 chunk_size,
                 embed_dim=300,
                 bi_lstm_units=256,
                 max_num_words=20000,
                 dropout_rate=0.1,
                 optimizer=Adam(),
                 emb_matrix=None,
                 weights_path=None,
                 src_tokenizer=None,
                 tgt_tokenizer=None):
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        self.max_num_words = max_num_words
        self.bi_lstm_units = bi_lstm_units
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.model = self.__build_model(emb_matrix)
        if weights_path is not None:
            try:
                self.model.load_weights(weights_path)
            except:
                print("Not weights found, create a new model.")

    def __build_model(self, emb_matrix=None):
        num_words = min(self.max_num_words, self.vocab_size)
        word_input = Input(shape=(None,), dtype='int32', name="word_input")

        if emb_matrix is not None:
            word_embedding = Embedding(num_words, self.embed_dim,
                                       weights=[emb_matrix],
                                       trainable=False,
                                       name='word_emb')(word_input)
        else:
            word_embedding = Embedding(num_words, self.embed_dim, name="word_emb") \
                (word_input)
        bilstm = Bidirectional(LSTM(self.bi_lstm_units // 2, return_sequences=True))(word_embedding)
        x = Dropout(self.dropout_rate)(bilstm)
        dense = Dense(self.chunk_size + 1)(x)
        crf = CRF(self.chunk_size + 1, sparse_target=False)
        crf_output = crf(dense)

        model = Model([word_input], [crf_output])

        model.compile(optimizer=self.optimizer, loss=crf.loss_function, metrics=[crf.accuracy])
        return model

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "chunk_size": self.chunk_size,
            "embed_dim": self.embed_dim,
            "bi_lstm_units": self.bi_lstm_units,
            "max_num_words": self.max_num_words,
            "dropout_rate": self.dropout_rate
        }

    @staticmethod
    def get_or_create(config, src_dict_path=None,
                      tgt_dict_path=None,
                      weights_path=None,
                      optimizer=Adam(),
                      encoding="utf-8"):
        if DLTokenizer.__singleton is None:
            if type(config) == str:
                with open(config, encoding=encoding) as file:
                    config = dict(json.load(file))
            elif type(config) == dict:
                config = config
            else:
                raise ValueError("Unexpect config type!")

            if src_dict_path is not None:
                config['src_tokenizer'] = load_dictionary(src_dict_path, encoding)
            if tgt_dict_path is not None:
                config['tgt_tokenizer'] = load_dictionary(tgt_dict_path, encoding)

            config['weights_path'] = weights_path
            config['optimizer'] = optimizer
            DLTokenizer.__singleton = DLTokenizer(**config)
        return DLTokenizer.__singleton

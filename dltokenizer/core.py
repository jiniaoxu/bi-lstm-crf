from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dropout, Dense, LSTM, Lambda
from keras.optimizers import Adam
from keras_contrib.layers import CRF


class DLTokenizer:

    def __init__(self,
                 vocab_size,
                 chunk_size,
                 embed_dim=300,
                 bi_lstm_units=256,
                 max_num_words=20000,
                 dropout_rate=0.1,
                 emb_matrix=None):
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim
        self.max_num_words = max_num_words
        self.bi_lstm_units = bi_lstm_units
        self.dropout_rate = dropout_rate
        self.model = self.__build_model(emb_matrix)

    def __build_model(self, emb_matrix=None):
        words_input = Input(shape=(None,), dtype='int32', name="words_input")
        chunk_labels = Input(shape=(None, self.chunk_size), dtype='float32', name='chunk_labels')

        if emb_matrix is not None:
            word_embedding = Embedding(self.vocab_size + 1, self.embed_dim,
                                       weights=[emb_matrix],
                                       trainable=True,
                                       name='word_emb')(words_input)
        else:
            word_embedding = \
                Embedding(self.vocab_size + 1, self.embed_dim, name="word_emb")(words_input)
        bilstm = Bidirectional(LSTM(self.bi_lstm_units // 2, return_sequences=True))(word_embedding)
        x = Dropout(self.dropout_rate)(bilstm)
        dense_chunk = Dense(self.chunk_size)(x)  # [N, -1, chunk_size]

        crf = CRF(self.chunk_size, sparse_target=False)
        crf_output = crf(dense_chunk)  # [N, -1, chunk_size]

        crf_loss = Lambda(lambda x: crf.loss_function(x[0], x[1]), name="crf_loss")([chunk_labels, crf_output])
        crf_accuracy = Lambda(lambda x: crf.accuracy(x[0], x[1]), name="crf_accuracy")([chunk_labels, crf_output])

        model = Model([words_input, chunk_labels], crf_loss)
        model.add_loss(crf_loss)

        model.compile(Adam(), None)
        model.metrics_names.append("crf_accuracy")
        model.metrics_tensors.append(crf_accuracy)

        return model

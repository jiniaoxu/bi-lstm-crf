from bi_lstm_crf_model import *
import numpy as np
from keras.utils import np_utils

model = BiLSTMCRFModelConfigure(1000, 4).build_model()

# train
x_train = np.random.randint(0, 1000, (500, 100))
y_train = np.random.randint(0, 4, (500, 100))
y_train = np_utils.to_categorical(y_train, 4)

print(x_train.shape)
print(y_train.shape)

model.fit(x_train, y_train, batch_size=16, epochs=2, verbose=1)

# test
test_data = np.random.randint(0, 1000, (10, 100))
test_y_pred = model.predict(test_data)
print(test_y_pred)
print(np.argmax(test_y_pred, axis=2))

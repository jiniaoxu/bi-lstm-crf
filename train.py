from keras.callbacks import ModelCheckpoint

from bi_lstm_crf_model import *
from process_data import *
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="进行模型的训练。")
    parse.add_argument("data_path", help="训练数据文件路径", default="./data/train.data")
    parse.add_argument("val_split", type=float, help="验证集所占比例。", default=0.2)
    parse.add_argument("save_dir", help="模型保存目录", default="./model")

    args = parse.parse_args()
    data_path = args.data_path
    val_split = args.val_split
    save_dir = args.save_dir

    x, y, word_index, _ = process_data(data_path)

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]
    num_validation_samples = int(val_split * x.shape[0])

    x_train = x[: -num_validation_samples]
    y_train = y[: -num_validation_samples]
    x_val = x[-num_validation_samples:]
    y_val = y[-num_validation_samples:]

    model_configure = BiLSTMCRFModelConfigure(word_index)
    model = model_configure.build_model()
    model.summary()

    ck = ModelCheckpoint(os.path.join(save_dir, 'weights.{epoch:02d}-{val_loss:.2f}.h5')
                         , monitor='loss', verbose=0)

    model.fit(x_train, y_train, batch_size=128, epochs=3,
              validation_data=(x_val, y_val), callbacks=[ck])

    save_model(model_configure
               , os.path.join(save_dir, 'model.cfg')
               , model
               , os.path.join(save_dir, 'crf.h5'))

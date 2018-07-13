from keras.callbacks import ModelCheckpoint

from bi_lstm_crf_model import *
from process_data import *
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="进行模型的训练。")
    parse.add_argument("--data_path", help="训练数据文件路径", default="./data/train.data")
    parse.add_argument("--val_split", type=float, help="验证集所占比例", default=0.2)
    parse.add_argument("--save_dir", help="模型保存目录", default="./model")
    parse.add_argument("--embedding_file_path", help="词嵌入文件路径，若不指定，则会训练词向量", default="")

    args = parse.parse_args()
    data_path = args.data_path
    val_split = args.val_split
    save_dir = args.save_dir
    embedding_file_path = None if args.embedding_file_path == "" else args.embedding_file_path

    x, y, word_index, chunk_index = process_data(data_path)

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]
    num_validation_samples = int(val_split * x.shape[0])

    x_train = x[: -num_validation_samples]
    y_train = y[: -num_validation_samples]
    x_val = x[-num_validation_samples:]
    y_val = y[-num_validation_samples:]

    model_configure = BiLSTMCRFModelConfigure(len(word_index) + 1, len(chunk_index) + 1)

    # 载入词向量
    embedding_matrix = None
    if embedding_file_path is not None:
        embedding_matrix = create_embedding_matrix(get_embedding_index(embedding_file_path), word_index,
                                                   model_configure)

    model = model_configure.build_model(embedding_matrix)
    model.summary()

    ck = ModelCheckpoint(os.path.join(save_dir, 'weights.{epoch:02d}-{val_loss:.2f}.h5')
                         , monitor='loss', verbose=0)

    model.fit(x_train, y_train, batch_size=64, epochs=10,
              validation_data=(x_val, y_val), callbacks=[ck])

    save_model(model_configure
               , os.path.join(save_dir, 'model.cfg')
               , model
               , os.path.join(save_dir, 'model.final.h5')
               , (word_index, chunk_index)
               , os.path.join(save_dir, 'model.dict'))

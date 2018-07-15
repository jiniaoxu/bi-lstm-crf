from keras.callbacks import ModelCheckpoint

from bi_lstm_crf_model import *
from process_data import *
import numpy as np
import argparse
import os


def random_split(x, y, split=0.2):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]
    num_validation_samples = int(split * x.shape[0])

    x_train = x[: -num_validation_samples]
    y_train = y[: -num_validation_samples]
    x_val = x[-num_validation_samples:]
    y_val = y[-num_validation_samples:]
    return x_train, y_train, x_val, y_val


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="进行模型的训练。")
    parse.add_argument("--data_path", help="训练数据文件路径", default="./data/train.data")
    parse.add_argument("--val_split", type=float, help="验证集所占比例", default=0.2)
    parse.add_argument("--save_dir", help="模型保存目录", default="./model")
    parse.add_argument("--model_dir", help="指定预训练的模型文件路径", default=None)
    parse.add_argument("--epochs", help="指定迭代几次", type=int, default=10)
    parse.add_argument("--embedding_file_path", help="词嵌入文件路径，若不指定，则会随机初始化词向量", default=None)

    args = parse.parse_args()
    DATA_PATH = args.data_path
    VAL_SPLIT = args.val_split
    SAVE_DIR = args.save_dir
    EMBEDDING_FILE_PATH = args.embedding_file_path
    MODEL_DIR = args.model_dir
    EPOCHS = args.epochs

    x, y, word_index, chunk_index = process_data(DATA_PATH)
    x_train, y_train, x_val, y_val = random_split(x, y, VAL_SPLIT)

    if MODEL_DIR is not None:
        word_index, chunk_index = load_dict(os.path.join(MODEL_DIR, "model.dict"))
        model_configure = load_model_config(os.path.join(MODEL_DIR, "model.cfg"))
        model = model_configure.build_model()
        model.load_weights(os.path.join(MODEL_DIR, "model.final.h5"))
    else:
        model_configure = BiLSTMCRFModelConfigure(len(word_index) + 1, len(chunk_index) + 1)
        # 保存模型配置
        save_model_config(model_configure, os.path.join(SAVE_DIR, 'model.cfg'))
        # 保存词汇表
        save_dict((word_index, chunk_index), os.path.join(SAVE_DIR, 'model.dict'))
        # 载入词向量
        embedding_matrix = None
        if EMBEDDING_FILE_PATH is not None:
            embedding_matrix = create_embedding_matrix(get_embedding_index(EMBEDDING_FILE_PATH), word_index,
                                                       model_configure)

        model = model_configure.build_model(embedding_matrix)

    model.summary()

    ck = ModelCheckpoint(os.path.join(SAVE_DIR, 'weights.{epoch:02d}-{val_loss:.2f}.h5')
                         , monitor='loss', verbose=0)

    model.fit(x_train, y_train, batch_size=512, epochs=EPOCHS,
              validation_data=(x_val, y_val), callbacks=[ck])

    # 保存最终的权重
    model.save(os.path.join(SAVE_DIR, 'model.final.h5'))

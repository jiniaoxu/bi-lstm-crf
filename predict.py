from bi_lstm_crf_model import *
import os


def predict(model: Model, chunk_index, sentence):
    pass


if __name__ == '__main__':
    model_base_dir = "./model"
    weights_path = os.path.join(model_base_dir, "model.final.h5")
    config_path = os.path.join(model_base_dir, "model.cfg")
    model, word_index, chunk_index = load_model(weights_path, config_path)

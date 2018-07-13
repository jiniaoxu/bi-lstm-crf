from bi_lstm_crf_model import *
from process_data import *
import os


def predict(sentence, word_index, model: Model, model_config: BiLSTMCRFModelConfigure):
    x = sentence_to_vec(sentence, word_index, model_config)
    preds = model.predict(x)
    return preds


def cut_sentence(sentence, tags):
    words = list(sentence)
    cuts, t1 = [], []
    for i, tag in enumerate(tags):
        if tag == 'B':
            if len(t1) != 0:
                cuts.append(t1)
            t1 = [words[i]]
        if tag == 'I':
            t1.append(words[i])
        if tag == 'S':
            if len(t1) != 0:
                cuts.append(t1)
            cuts.append([words[i]])
            t1 = []
        if i == len(tags) - 1 and len(t1) != 0:
            cuts.append(t1)
    return cuts


if __name__ == '__main__':
    model_base_dir = "./model"
    weights_path = os.path.join(model_base_dir, "model.final.h5")
    config_path = os.path.join(model_base_dir, "model.cfg")
    dict_path = os.path.join(model_base_dir, "model.dict")
    model, config, word_index, chunk_index = load_model(weights_path, config_path, dict_path)
    sentence = "历经重重考验之后，火炬终于传到了他的手中"
    preds = predict(sentence, word_index, model, config)[0][-len(sentence):]
    result = np.argmax(preds, axis=1)
    index_chunk = {i: c for c, i in chunk_index.items()}
    result_tag = list(map(lambda i: index_chunk[i], result))
    print(sentence)
    print(result_tag)
    print(cut_sentence(sentence, result_tag))

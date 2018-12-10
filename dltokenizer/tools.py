import json

import pandas as pd
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json


def _parse_data(fh):
    text = fh.readlines()
    sent, t1, t2, chunk = [], [], [], []
    for i in text:
        if i != '\n':
            char, tag = i.split()
            t1.append(char)
            t2.append(tag)
        elif len(t1) != 0:
            sent.append(t1)
            chunk.append(t2)
            t1, t2 = [], []
    fh.close()
    sent = pd.Series(sent)
    chunk = pd.Series(chunk)
    return sent, chunk


def save_dictionary(tokenizer, dict_path, encoding="utf-8"):
    with open(dict_path, mode="w+", encoding=encoding) as file:
        json.dump(tokenizer.to_json(), file)


def load_dictionary(dict_path, encoding="utf-8"):
    with open(dict_path, mode="r", encoding=encoding) as file:
        return tokenizer_from_json(json.load(file))


def make_dictionaries(file_path,
                      src_dict_path=None,
                      tgt_dict_path=None,
                      encoding="utf-8",
                      min_feq=5,
                      **kwargs):
    sents, chunks = _parse_data(open(file_path, 'r', encoding=encoding))
    src_tokenizer = Tokenizer(**kwargs)
    tgt_tokenizer = Tokenizer(**kwargs)

    src_tokenizer.fit_on_texts(sents)
    tgt_tokenizer.fit_on_texts(chunks)

    src_sub = sum(map(lambda x: x[1] < min_feq, src_tokenizer.word_counts.items()))
    tgt_sub = sum(map(lambda x: x[1] < min_feq, tgt_tokenizer.word_counts.items()))

    src_tokenizer.num_words = len(src_tokenizer.word_index) - src_sub
    tgt_tokenizer.num_words = len(tgt_tokenizer.word_index) - tgt_sub

    if src_dict_path is not None:
        save_dictionary(src_tokenizer, src_dict_path, encoding=encoding)
    if tgt_dict_path is not None:
        save_dictionary(tgt_tokenizer, tgt_dict_path, encoding=encoding)

    return src_tokenizer, tgt_tokenizer


if __name__ == '__main__':
    src_tokenizer, tgt_tokenizer = make_dictionaries("../corups/bis.txt", filters='')
    print(src_tokenizer.num_words)
    print(tgt_tokenizer.num_words)

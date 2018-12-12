from dltokenizer.tools import make_dictionaries, load_dictionaries

if __name__ == '__main__':
    make_dictionaries("../data/2014",
                      "../data/src_dict.json",
                      "../data/tgt_dict.json",
                      min_feq=5,
                      oov_token="<UNK>",
                      filters='')

    src_tokenizer, tgt_tokenizer = load_dictionaries("../data/src_dict.json",
                                                     "../data/tgt_dict.json")
    print(src_tokenizer.texts_to_sequences(["万 科 陷 入 “ 欠 稅 风 波 ” "]))
    print(tgt_tokenizer.num_words)

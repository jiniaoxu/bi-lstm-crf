import os


class DataLoader:

    def __init__(self,
                 src_dict_path,
                 tgt_dict_path,
                 batch_size=64,
                 max_len=999,
                 total_size=441959,
                 encoding="utf-8"):
        # self.src_tokenizer = load_dictionary(src_dict_path, encoding)
        # self.tgt_tokenizer = load_dictionary(tgt_dict_path, encoding)
        self.batch_size = batch_size
        self.max_len = max_len
        self.steps_per_epoch = total_size // self.batch_size
        self.total_size = total_size

    def generator(self, file_path, encoding="utf-8"):
        if os.path.isdir(file_path):
            while True:
                for sent, chunk in self.load_sents_from_dir(file_path):
                    yield sent, chunk
        while True:
            for sent, chunk in self.load_sents_from_file(file_path, encoding):
                yield sent, chunk

    def load_sents_from_dir(self, source_dir, encoding="utf-8"):
        for root, dirs, files in os.walk(source_dir):
            for name in files:
                file = os.path.join(root, name)
                for sent, chunk in self.load_sents_from_file(file, encoding=encoding):
                    yield sent, chunk

    def load_sents_from_file(self, file_path, encoding):
        with open(file_path, encoding=encoding) as f:
            sent, t1, t2, chunk = [], [], [], []
            for line in f:
                if line != '\n':
                    char, tag = line.split()
                    t1.append(char)
                    t2.append(tag)
                elif len(t1) != 0:
                    sent.append(t1)
                    chunk.append(t2)
                    t1, t2 = [], []
                if len(sent) >= self.batch_size:
                    # sent = pd.Series(sent)
                    # chunk = pd.Series(chunk)
                    yield sent, chunk
                    sent, chunk = [], []


if __name__ == '__main__':
    data_loader = DataLoader(None, None, batch_size=128)

    generator = data_loader.generator("../data/2014")
    for _ in range(3):
        sent, chunk = next(generator)
        print(len(sent))
        assert len(sent) == len(chunk)
        print(sent)
        print(chunk)

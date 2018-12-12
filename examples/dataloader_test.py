from dltokenizer.data_loader import DataLoader

if __name__ == '__main__':
    data_loader = DataLoader("../data/src_dict.json", "../data/tgt_dict.json", batch_size=64)

    generator = data_loader.generator("../data/2014")
    for _ in range(1):
        sent, chunk = next(generator)
        assert len(sent) == len(chunk)
        print(sent.shape)
        print(chunk.shape)

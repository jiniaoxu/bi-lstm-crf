import multiprocessing
import time
import argparse

from rx import Observable
from rx.concurrency import ThreadPoolScheduler

from bi_lstm_crf_model import *
from process_data import *


def predict(sentences, word_index, index_chunk, model: Model, model_config: BiLSTMCRFModelConfigure,
            parallel=False) -> Observable:
    x = sentence_to_vec(sentences, word_index, model_config)

    start = time.clock()
    preds = model.predict(x, batch_size=1024)
    print("Predict cost time {} s".format(time.clock() - start))
    tags_encode = np.argmax(preds, axis=2)
    tags_decode = Observable.of(*tags_encode)

    if parallel:
        return Observable.zip(Observable.of(*sentences), tags_decode, lambda s, i: (s, i)) \
            .flat_map(lambda v: Observable.just(v)
                      .subscribe_on(pool_scheduler)
                      .map(lambda v: (v[0], v[1][-len(v[0]):]))
                      .map(lambda v: (v[0], list(map(lambda i: index_chunk[i], v[1]))))
                      .map(lambda v: cut_sentence_str(*v)))
    else:
        return Observable.zip(Observable.of(*sentences), tags_decode, lambda s, i: (s, i)) \
            .map(lambda v: (v[0], v[1][-len(v[0]):])) \
            .map(lambda v: (v[0], list(map(lambda i: index_chunk[i], v[1])))) \
            .map(lambda v: cut_sentence_str(*v))


def cut_sentence(sentence, tags):
    words = list(sentence)
    cuts, t1 = [], []
    for i, tag in enumerate(tags):
        if tag == 'B':
            if len(t1) != 0:
                cuts.append(t1)
            t1 = [words[i]]
        elif tag == 'I':
            t1.append(words[i])
        elif tag == 'S':
            if len(t1) != 0:
                cuts.append(t1)
            cuts.append([words[i]])
            t1 = []
        if i == len(tags) - 1 and len(t1) != 0:
            cuts.append(t1)
    return cuts


def cut_sentence_str(sentence, tags):
    cuts = cut_sentence(sentence, tags)
    words = list(map(lambda word_cuts: ''.join(word_cuts), cuts))
    return words


def _load_sentences(text_file, max_sentence_len):
    with open(text_file, "r", encoding="UTF-8") as f:
        return list(map(lambda line: line[:min(len(line), max_sentence_len)], f.readlines()))


def _save_pred(pred, pred_file_path):
    stream = pred.reduce(lambda a, b: a + b)
    with open(pred_file_path, "a", encoding="UTF-8") as f:
        stream.subscribe(lambda text: f.write(' '.join(text)))


if __name__ == '__main__':
    optimal_thread_count = multiprocessing.cpu_count()
    pool_scheduler = ThreadPoolScheduler(optimal_thread_count)

    parser = argparse.ArgumentParser(description="分割一段或几段文本，每段文本不超过150词，否则会截断。")
    parser.add_argument("-m", "--model_dir", help="指定模型目录", default="./model")
    parser.add_argument("-pf", "--pref_file_path", help="将分词结果保存到指定文件中", default=None)

    parser.add_mutually_exclusive_group()
    parser.add_argument("-s", "--sentence", help="指定要分词的语句，以空格' '分割多句")
    parser.add_argument("-tf", "--text_file_path", help="要分割的文本文件的路径，文本中每一行为一句话。")

    args = parser.parse_args()

    model_base_dir = args.model_dir
    config = load_model_config(os.path.join(model_base_dir, "model.cfg"))
    word_index, chunk_index = load_dict(os.path.join(model_base_dir, "model.dict"))
    model = config.build_model()
    model.load_weights(os.path.join(model_base_dir, "model.final.h5"))

    index_chunk = {i: c for c, i in chunk_index.items()}

    if args.sentence:
        sentences = args.sentence.split()
    elif args.text_file_path:
        sentences = _load_sentences(args.text_file_path, config.max_sequence_len)
    else:
        raise RuntimeError("你必须通过-s 获 -tf 选项指定要进行分词的文本。")

    # sentences = [
    #     "Multi-tasklearning （多任务学习）是和single-task learning （单任务学习）相对的一种机器学习方法。",
    #     "拿大家经常使用的school data做个简单的对比，school data是用来预测学生成绩的回归问题的数据集，总共有139个中学的15362个学生，其中每一个中学都可以看作是一个预测任务。",
    #     "单任务学习就是忽略任务之间可能存在的关系分别学习139个回归函数进行分数的预测，或者直接将139个学校的所有数据放到一起学习一个回归函数进行预测。"]

    result = predict(sentences, word_index, index_chunk, model, config)

    if args.pref_file_path:
        _save_pred(result, args.pref_file_path)
    else:
        result.subscribe(lambda v: print(v))

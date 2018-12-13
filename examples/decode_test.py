import time

from dltokenizer import get_or_create

if __name__ == '__main__':
    tokenizer = get_or_create("../data/default-config.json",
                              src_dict_path="../data/src_dict.json",
                              tgt_dict_path="../data/tgt_dict.json",
                              weights_path="../models/weights.14-0.05.sgdr.h5")

    for _ in range(1):
        start_time = time.time()
        for sent, tag in tokenizer.decode_texts([
            "美国司法部副部长罗森·施泰因（Rod Rosenstein）指，"
            "这些俄罗斯情报人员涉嫌利用电脑病毒或“钓鱼电邮”，"
            "成功入侵民主党的电脑系统，偷取民主党高层成员之间的电邮，"
            "另外也从美国一个州的电脑系统偷取了50万名美国选民的资料。"
        ]):
            print(sent)
            print(tag)
        print(f"cost {(time.time() - start_time) * 1000}ms")

# bi-lstm-crf

## 简介

不同于英文自然语言处理，中文自然语言处理，例如语义分析、文本分类、词语蕴含等任务都需要预先进行分词。要将中文进行分割，直观的方式是通过为语句中的每一个字进行标记，以确定这个字是位于一个词的开头还是之中：

例如“**成功入侵民主党的电脑系统**”这句话，我们为其标注为：

```js
"成功 入侵  民主党 的 电脑系统"
 B I  B I  B I I  S  B I I I
```

其中`B`表示一个词语的开头，`I`表示非一个词语的开头，`S`表示单字成词。这样我们就能达到分词的效果。

对于句子这样的序列而言，要为其进行标注，常用的是使用Bi-LSTM卷积网络进行序列标注，如下图：

<center>
    <img src="./imgs/Bi-LSTM.png" width=400 title="Bi-LSTM 的序列标注">
    </img>
</center>

通过Bi-LSTM获得每个词所对应的所有标签的概率，取最大概率的标注即可获得整个标注序列，如上图序列`W0W1W2`的标注为`BIS`。但这样有可能会取得不合逻辑的标注序列，如`BS`、`SI`等。我们需要为其设定一些约束，如：

* B后只能是I
* S之后只能是B、S
* ...

而要做到这一点，我们可以在原有的模型基础之上，加上一个CRF层，该层的作用即是学习符号之间的约束（如上所述）。模型架构变为Embedding + Bi-LSTM + CRF，原理参考论文：https://arxiv.org/abs/1508.01991。

## 语料预处理

要训练模型，首先需要准备好语料，这里选用人民日报2014年的80万语料作为训练语料。语料格式如下：

```js
"人民网/nz 1月1日/t 讯/ng 据/p 《/w [纽约/nsf 时报/n]/nz 》/w 报道/v ，/w 美国/nsf 华尔街/nsf 股市/n 在/p 2013年/t 的/ude1 最后/f 一天/mq 继续/v 上涨/vn ，/w 和/cc [全球/n 股市/n]/nz 一样/uyy ，/w 都/d 以/p [最高/a 纪录/n]/nz 或/c 接近/v [最高/a 纪录/n]/nz 结束/v 本/rz 年/qt 的/ude1 交易/vn 。/w "
```

每一个词语使用空格分开后面使用POS标记词性，而本模型所需要的语料格式如下：

```js
嫌 疑 人 赵 国 军 。    B-N I-N I-N B-NR I-NR I-NR S-W
```

使用命令:

```sh
python dltokenizer/data_preprocess.py <语料目录> <输出目录> 
```

可将原始的有词性标注的文档转换为使BIS（B:表示语句块的开始，I:表示非语句块的开始，S:表示单独成词）标注的文件。

## 使用

### 生成字典

```python
examples/dict_test.py
```

默认将会使用`data/2014`文件夹下的文件（已转为BIS格式），生成两个字典文件，`data/src_dict.json`, `data/tgt_dict.json`

### 训练

```python
examples/train_test.py
```

训练时，默认会生成模型配置文件`data/default-config.json`, 权重文件将会生成在`models`文件夹下。

#### 使用字（词）向量

在训练时可以使用已训练的字（词）向量作为每一个字的表征，字（词）向量的格式如下：

```js
而 -0.037438 0.143471 0.391358 ...
个 -0.045985 -0.065485 0.251576 ...
以 -0.085605 0.081578 0.227135 ...
可以 0.012544 0.069829 0.117207 ...
第 -0.321195 0.065808 0.089396 ...
上 -0.186070 0.189417 0.265060 ...
之 0.037873 0.075681 0.239715 ...
于 -0.197969 0.018578 0.233496 ...
对 -0.115746 -0.025029 -0 ...
```

每一行，为一个字（词）和它所对应的特征向量。

汉字字（词）向量来源
可从[https://github.com/Embedding/Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors)获得字（词）向量。字（词）向量文件中每一行格式为一个字（词）与其对应的300维向量。

### 分词/解码

1. 编码方式：

    ```python
    import time

    from dltokenizer import get_or_create

    if __name__ == '__main__':
        tokenizer = get_or_create("../data/default-config.json",
                                src_dict_path="../data/src_dict.json",
                                tgt_dict_path="../data/tgt_dict.json",
                                weights_path="../models/weights.14-0.15.h5")

        for _ in range(1):
            start_time = time.time()
            for sent, tag in tokenizer.decode_texts([
                "美国司法部副部长罗森·施泰因（Rod Rosenstein）指，"
                "这些俄罗斯情报人员涉嫌利用电脑病毒或“钓鱼电邮”，"
                "成功入侵民主党的电脑系统，偷取民主党高层成员之间的电邮，"
                "另外也从美国一个州的电脑系统偷取了50万名美国选民的资料。"]):
                print(sent)
                print(tag)
            print(f"cost {(time.time() - start_time) * 1000}ms")

    ```

    `get_or_create`接受4个参数：

    * config_path: 模型配置路径
    * src_dict_path：源字典文件路径
    * tgt_dict_path：目标字典文件路径
    * weights_path：权重文件路径

2. 命令方式：

    ```python
    python examples/predict.py -s <语句>
    ```

    命令方式所使用的模型配置文件、字典文件等如编程方式中所示。进行分词时，多句话可用空格分隔，具体使用方式可使用`predict.py -h`查看。

### 分词效果展示

待续。。。

### 训练效果

待续。。。

## 其它

### 如何评估

使用与黄金标准文件进行对比的方式，进行评估。

1. 数据预处理

    为了生成黄金标准文件和无分词标记的原始文件，可用下列命令：

    ```python
    python examples/score_preprocess.py --corups_dir <语料文件夹> \
    --gold_file_path <生成的黄金标准文件路径> \
    --restore_file_path <生成无标记的原始文件路径>
    ```

2. 读取无标记的原始文件，并进行分词，输出到文件：

    ```python
    python examples/predict.py -f <要分割的文本文件的路径> -o <保存分词结果的文件路径>
    ```

3. 生成评估结果：

    执行`score.py`可生成评估文件，默认使用黄金分割文件`./score/gold.utf8`，使用模型分词后的文件`./score/pred_text.utf8`，评估结果保存到`prf_tmp.txt`中。

    ```py
    def main():
        F = prf_score('./score/gold.utf8', './score/pred_text.utf8', 'prf_tmp.txt', 15)
    ```

## 附录

1. 分词语料库： https://pan.baidu.com/s/1-LLzKOJglP5W0VCVsI0efg 密码: krhr
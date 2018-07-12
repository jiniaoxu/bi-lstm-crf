# bi-lstm-crf

## 简介

采用序列标注的方式实现中文分词，模型架构为Embedding + Bi-LSTM + CRF，参考论文：https://arxiv.org/abs/1508.01991。

## 使用方式

### 数据预处理

原始的语料为人民日报的 80 万语料（语料库很大，可在附录获得）。
这些语料已经对语句进行了切分和词性标注，为了转换为模型可用的语料，使用命令:
```python
preprocess_data.py <语料目录> train.data -a
```
将原始的有词性标注的文档转换为使BIS（B:表示语句块的开始，I:表示非语句块的开始，S:表示单独成词）标注的文件。

### 训练

使用`train.py`进行模型的训练

## 附录

1. 语料库： https://pan.baidu.com/s/1-LLzKOJglP5W0VCVsI0efg 密码: krhr 

# bi-lstm-crf

## 简介



## 使用方式

1. 数据预处理

使用`preprocess_data.py <语料目录> train.data`将原始的有词性标注的文档转换为使用BIS（B:表示语句块的开始，I:表示非语句块的开始，S:表示单独成词）标注的文件

2. 训练

使用`train.py`进行模型的训练

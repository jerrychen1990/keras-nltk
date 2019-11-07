# A toolkit for nlp job in eigen
开发了一些信息抽取的任务之后，发现除了模型的调优，有很多预处理以及后处理的工作都是比较麻烦且费时的（比如NER任务中不同语言经过bert tokenize 之后，word piece与原文的对应问题等），同时这些工作也是通用的、大家都会遇到的。 eigen-nlp-toolkit项目主要为了解决这些问题。本项目是基于keras框架的更高层次的封装，直接面向具体的nlp任务（目前包括NER,Classify,NRE,MLM-pretrain）。应用本框架，可以基于配置的做以上四类任务，可以方便的在yud或者pony上做实验。这四类任务的输入都是json格式的，输出包括模型文件、评价指标文件、数据结果文件(用来做错误分析)。具体操作方式可以看考examples文件夹下的四个example。除此之外，本工具包还提供了一些基于自己改进的optimizer, callback function, data transform function等。

下面针对项目下的不同python module文件一一做介绍：


## install
```bash
pip install --upgrade pip
pip install eigen_nltk
```
注：需要将pip安装源设置为 https://pypi-outer.aidigger.com/simple

## core
定义了一些基础类
- Context: 一些上下文信息，主要包括词表, ner任务包括ner标签字典, classify任务包括label字典等
- BaseEstimator: 定义了predict_batch接口,可以做预测, 也可以方便的和euler需要的SimplexBaseModel对接
- ModelEstimator: 继承BaseEstimator,包含一个keras模型，在其上作了数据预处理、后处理等工作
- RuleEstimator: 继承BaseEstimator，不包含模型，通过规则做预测
- EnsembleEstimator：继承BaseEstimator，内部可以包含多个ModelEstimator，只用作预测，不用做训练

## ner
NER相关任务的封装
- NerExtractor:继承自ModelEstimator, 可以端到端做ner任务
- EntityPairNerExtractor:继承自ModelEstimator，用entity1_start, entity1_end, entity2_start, entity2_end的方式标注两个实体。针对某些关系抽取任务做的定制，确保一个sequence一定识别出两个entity


主要的改进:
- 1.屏蔽了token和原文本的对齐问题，模型的输出包括原文中出现的实体名、以及在原文中的offset。
- 2.屏蔽了长文本拆分的问题，对于超过模型长度的文本的预测，会自动用比较合理的方式拆分文本，最后再将拆开之后模型的预测结果合并
- 3.增加了crf层的constraint参数，可以自定义NER tag之间的转移状态
- 4.增加了锁定bert某几层的参数，fine-tune bert的时候可以锁住最下面的N个transformer block（这个是所有基于bert的模型通用的）

example：
```bash
cd examples
python ner_example.py
```

## nre
NRE相关任务的封装。任务定义:给定一个sentence以及其中的两个实体entity1, entity2,输出这两个实体的关系类别。

参考[《Matching the Blanks- Distributional Similarity for Relation Learning》](https://arxiv.org/abs/1906.03158)的实验，采用在两个entity前后加上特殊的tag token，再把tag token的embedding concat起来做分类预测的方案
主要改进和NER类似

example：
```bash
cd examples
python nre_example.py
```

## classify
比较简单用bert结果的[CLS]或者根据LSTM的最后一个隐特征做多分类
参考 [《How to Fine-Tune BERT for Text Classiﬁcation_》](https://arxiv.org/abs/1905.05583)
主要改进和NER类似

example：
```bash
cd examples
python classify_example.py
```

## pretrain
针对领域相关的预料，对bert进行further pretrain。 可以基于发布的bert模型做fine-tune，也可以自己构建一个小型的bert，重新训练
example：
```bash
cd examples
python pretrain_example.py
```

## optimizer
针对keras的两个优化器Adam和AdamWarmup做了针对bert模型的优化，可以让bert模型在fine-tune的时候，学习率从高层到底层按照decay_rate的比例递减。实现的两个优化器为BertAdam和BertAdamWarmup

## eval
针对四个任务，作了通用的评价指标，同时实现了对于每个sample的gold data 与predict data的对比结果记录。

## callback
实现一个keras训练时的callback函数，每个epoch时对dev集合做测评，根据eval模块返回的评价指标，对比改epoch是否获得了当前最好的指标，如果是，则保存模型

## experiment
训练一个模型的工作流， 可以通过配置文件的方式控制训练流程。应该可以比较方面的加入到pony系统中做成plugin。 每个experiment结束后，模型文件会保存到model/文件夹下，评价指标在eval/文件夹，详细输出结果的对比在output/文件夹下

## trans
做数据转化的函数

## utils
一些工具函数 




# Introduction

The purpose of create this repository is to learn [Transformer](https://arxiv.org/abs/1706.03762).

most code clone from [Tensorflow-models](https://github.com/tensorflow/models/tree/master/official/transformer)

## What I did ?

- split vocab to inputs_vocab and targets_vocab
- delete some tpu config
- cut chinese to words 

## run demo

1. create vocab and tfrecord
```python
python process_data.py
```

2. train_model
```python
./en_zh_train.sh
```

you can change ```en_zh_train.sh``` to test and export model


There is an [demo](ai.midday.me)


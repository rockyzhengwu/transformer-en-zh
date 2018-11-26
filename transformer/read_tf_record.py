#-*-coding:utf-8-*-

import tensorflow as tf
import os

base_dir = './train_data'

filenames = []
for top, dirs, files in os.walk(base_dir):
    for name in files:
        if 'dev' in name:
            filenames.append(os.path.join(top, name))
data_set = tf.data.TFRecordDataset(filenames=filenames)

def parse_example(serialized_example):
  """Return inputs and targets Tensors from a serialized tf.Example."""
  data_fields = {
      "inputs": tf.VarLenFeature(tf.int64),
      "targets": tf.VarLenFeature(tf.int64)
  }
  parsed = tf.parse_single_example(serialized_example, data_fields)
  inputs = tf.sparse_tensor_to_dense(parsed["inputs"])
  targets = tf.sparse_tensor_to_dense(parsed["targets"])
  return inputs, targets



data_set = data_set.map(parse_example)
data_set = data_set.prefetch(2)
iterator = data_set.make_one_shot_iterator()
sess = tf.Session()
i=0
while True:
    if i%1000 == 0:
        print(i)

    example = sess.run(iterator.get_next())
    print(example[0])
    print(example[1])
    si = [t for t in example[0] if not t]
    so = [t for t in example[1] if not t]
    assert len(si)==len(so)==0
    i+=1
    #print(example)
    #print(len(example[0]))
    #input()

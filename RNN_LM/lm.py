# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

from distutils.version import StrictVersion
flags = tf.flags
logging = tf.logging

def data_type():
  return  tf.float32

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, bos_id=0, eos_id=0, sample_weight=True):
    self._is_training = is_training
    self._rnn_params = None
    self._cell = None
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    self.data=tf.placeholder(shape=[None, None], dtype=tf.int32)
    self.data_len=tf.placeholder(shape=[None], dtype=tf.int32)
    if sample_weight:
      self.sample_weight=tf.placeholder(shape=[None], dtype=tf.float32)
    self.input=tf.concat([tf.zeros([tf.shape(self.data_len)[0]+bos_id, 1], dtype=tf.int32), self.data], axis=1)
    self.target=tf.concat([self.data, tf.zeros([tf.shape(self.data_len)[0]+bos_id, 1], dtype=tf.int32 )], axis=1)
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self.input)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    output, state = self._build_rnn_graph(inputs, config, is_training)

    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
     # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        self.target,
        tf.sequence_mask(self.data_len, self.num_steps, dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=False)
    loss_prefix = tf.contrib.seq2seq.sequence_loss(
        logits,
        self.target,
        tf.sequence_mask(self.data_len, self.num_steps, dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=False)

    # Update the cost
    self._cost = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    self._final_state = state
    self.NLL=tf.reduce_sum(loss)/(tf.reduce_sum(tf.cast(self.data_len, dtype=tf.float32))+1e-7)
    self.NLL_each=tf.reduce_sum(loss, axis=1)
    self.NLL_each_prefix=tf.reduce_sum(loss_prefix, axis=1)
    #if not is_training:
    #  return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    self.tvars=tvars
    ##Following part is for gradient calculation. Please set batchsize=1
    self.grads_vec=tf.concat([tf.reshape(x, [-1]) for x in tf.gradients(self._cost, tvars)], axis=0)
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training):
    return self._build_rnn_graph_lstm(inputs, config, is_training)

  def _get_lstm_cell(self, config, is_training):
    if True:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)

  def _build_rnn_graph_lstm(self, inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    # outputs, state = tf.nn.static_rnn(cell, inputs,
    #                                   initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    return output, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

class LM:
  def __init__(self, config, bos_id=0, eos_id=0, scope='LM'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      self.train_model=PTBModel(True, config, bos_id=0, eos_id=0)
      self.eval_model=PTBModel(False, config, bos_id=0, eos_id=0)
      self.lr=config.learning_rate
      self.scope=scope
      var_list=[x for x in tf.global_variables() if x.name.startswith(scope)]
      self.initializer=tf.variables_initializer(var_list=var_list)
      self.saver=tf.train.Saver(var_list=self.train_model.tvars, max_to_keep=0)
      self.grads_vec=self.eval_model.grads_vec
  def train_step(self, sess, data, data_len):
    model=self.train_model
    model.assign_lr(sess, self.lr)
    _, NLL=sess.run([model._train_op, model.NLL], feed_dict={model.data:data, model.data_len:data_len})
    return NLL
  def eval_step(self, sess, data, data_len):
    model=self.eval_model
    NLL=sess.run(model.NLL, feed_dict={model.data:data, model.data_len:data_len})
    return NLL
  def eval_each_step(self, sess, data, data_len, prefix=True):
    model=self.eval_model
    if prefix:
      NLL_each=sess.run(model.NLL_each_prefix, feed_dict={model.data:data, model.data_len:data_len})
    else:
      NLL_each=sess.run(model.NLL_each, feed_dict={model.data:data, model.data_len:data_len})
    return NLL_each
  def get_gradient(self, sess, data, data_len):
    model=self.eval_model
    return sess.run(model.grads_vec, feed_dict={model.data:data, model.data_len:data_len})
  def initialize(self, sess):
    sess.run(self.initializer)
  def save(self, sess, path, global_step=0):
    #path should be a dir
    self.saver.save(sess, path, global_step=global_step)
  def restore(self, sess, path, global_step=0):
    #path=tf.train.latest_checkpoint(path)
    self.saver.restore(sess, path+'-'+str(global_step))

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 10
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 21
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def get_config(model):
  """Get model config."""
  config = None
  if model == "small":
    config = SmallConfig()
  elif model == "medium":
    config = MediumConfig()
  elif model == "large":
    config = LargeConfig()
  elif model == "test":
    config = TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)
  return config

if __name__=='__main__':
  lm=LM(config=get_config('small'))
  sess=tf.Session()
  lm.initialize(sess)
  lm.save(sess, '/mnt/cephfs_new_wj/mlnlp/miaoning/Experiment/RNN_LM/models/')
  lm.restore(sess, '/mnt/cephfs_new_wj/mlnlp/miaoning/Experiment/RNN_LM/models/')
  data=np.random.randint(0,1000, [10,19], dtype=np.int32)
  data_len=np.ones(shape=[10], dtype=np.int32)*10
  for i in range(20):
    print(lm.train_step(sess, data, data_len))
  print(lm.eval_step(sess, data, data_len))
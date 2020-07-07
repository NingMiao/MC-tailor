from data import *
import argparse
from lm import *
import os

##Data loading
class args:
  vocab_path='./RNN_LM/text.voc'
  seq_len=30
  batch_size=128
  learning_rate=1e-3
  val_every=100
  val_batch_num=10000

Vocab=vocab(args.vocab_path)



class Config(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = args.learning_rate
  max_grad_norm = 5
  num_layers = 2
  num_steps = args.seq_len+1
  hidden_size = 256
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = args.batch_size
  vocab_size = Vocab.vocab_size

lm=LM(config=Config())
sess=tf.Session()
lm.initialize(sess)

def file_f(train_data_path, val_data_path, verbose=False, line_lim=0, word_lim=0):
  print('start calculating rev_ppl')
  lm.initialize(sess)
  val_NLL_mean_old=100.0
  flag=0
  train_dataset=dataset(Vocab, train_data_path, args.seq_len, line_lim, word_lim)
  val_dataset=dataset(Vocab, val_data_path, args.seq_len, 10000)
  for i in range(10000):
    train_batch, train_batch_len, _=train_dataset.get_batch(args.batch_size)
    if verbose:
      print(lm.train_step(sess, train_batch, train_batch_len))
    else:
      lm.train_step(sess, train_batch, train_batch_len)
    if i%args.val_every==0:
      val_NLL=[]
      for j in range(args.val_batch_num):
        val_batch, val_batch_len, pointer=val_dataset.get_batch(args.batch_size)
        val_NLL.append(lm.eval_step(sess, val_batch, val_batch_len))
        if pointer==0:
          break
      val_NLL_mean=np.mean(val_NLL)
      if verbose:
        print('val_NLL:{}'.format(val_NLL_mean))
      val_dataset.pointer=0
      if val_NLL_mean>=val_NLL_mean_old:
        flag+=1
        if flag>=5:
          return val_NLL_mean_old
      else:
        val_NLL_mean_old=val_NLL_mean
        flag=0
  return val_NLL_mean_old
    
if __name__=='__main__':
  print(file_f('./samples/finetune/sample2_3k.txt', './data/onto_bn/test.txt', word_lim=4000))
  print(file_f('./samples/discri/sample2_3k.txt', './data/onto_bn/test.txt', word_lim=4000))


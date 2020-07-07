#!/usr/bin/env python3
import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
import pickle as pkl
from tensorflow.core.protobuf import rewriter_config_pb2
from copy import deepcopy as copy

import sys
sys.path.insert(0, './src')
import model, sample, encoder
sys.path.insert(0, './RNN_LM')
import train

from load_dataset import load_dataset, Sampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients
from discri import *
CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'
sample_link=sample


parser = argparse.ArgumentParser(description='MC-tailor-ERS.')
#Data config
parser.add_argument('--data_dir', metavar='PATH', type=str, default='./data/', help='dictionary of data, see ./data for examples')
parser.add_argument('--dataset', type=str, default='onto_bn')

#GPT-2 config
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('--batch_size', metavar='SIZE', type=int, default=256, help='Batch size')
parser.add_argument('--seq_len', metavar='SIZE', type=int, default=30, help='')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00002, help='Learning rate for Adam')
parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1, help='Accumulate gradients across N minibatches.')
parser.add_argument('--memory_saving_gradients', default=False, action='store_true', help='Use gradient checkpointing to reduce vram usage.')
parser.add_argument('--only_train_transformer_layers', default=False, action='store_true', help='Restrict training to the transformer blocks.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer. <adam|sgd>.')
parser.add_argument('--noise', type=float, default=0.0, help='Add noise to input training data to regularize against typos.')

parser.add_argument('--top_k', type=int, default=40, help='K for top-k sampling.')
parser.add_argument('--top_p', type=float, default=0.0, help='P for top-p sampling. Overrides top_k if set > 0.')

#Tailor config
parser.add_argument('--layer_num', type=int, default=3, help='Layers of dis model')
parser.add_argument('--exponential_param', type=float, default=2.0)
parser.add_argument('--pos_loss_weight', type=float, default=2.5)

#Training config
parser.add_argument('--restore_from', type=str, default='fresh', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=10, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=1023, help='Sample this many tokens')
parser.add_argument('--sample_num', metavar='N', type=int, default=10, help='Generate this many samples')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=256, help='Batch size for validation.')
parser.add_argument('--val_batch_count', metavar='N', type=int, default=3, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=10, help='Calculate validation loss every STEPS steps.')

#Path
parser.add_argument('--gpt_save_dir', type=str, default='./models/gen_ERS/')  
parser.add_argument('--dis_save_dir', type=str, default='./models/dis_ERS/')
parser.add_argument('--gpt_sample_dir', type=str, default='./samples/gen_ERS/')
parser.add_argument('--dis_sample_dir', type=str, default='./samples/dis_ERS/')
parser.add_argument('--log_dir', type=str, default='./logs/log_ERS/')

#Action
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--evaluate_finetune', action='store_true')
parser.add_argument('--train_tailor', action='store_true')
parser.add_argument('--evaluate_tailor', action='store_true')

args = parser.parse_args()

def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context




def main():
    
    enc = encoder.get_encoder(args.model_name)
    hparams = model.default_hparams()
    hparams.batch_size=args.batch_size
    hparams.seq_len=args.seq_len
    
    ##data_path
    args.train_data_path=args.data_dir+args.dataset+'/train.txt'
    args.eval_data_path=args.data_dir+args.dataset+'/dev.txt'
    args.test_data_path=args.data_dir+args.dataset+'/test.txt'
    args.eval_data_path=args.test_data_path                          ###Test mode only!
    args.gpt_save_path=args.gpt_save_dir+args.dataset+'/'
    args.dis_save_path=args.dis_save_dir+args.dataset+'/'
    
    args.gpt_sample_dir2=args.gpt_sample_dir+args.dataset+'/'
    args.dis_sample_dir2=args.dis_sample_dir+args.dataset+'/'
    
    args.log_path=args.log_dir+args.dataset+'/'
    maketree(args.gpt_save_dir)
    maketree(args.dis_save_dir)
    maketree(args.gpt_save_path)
    maketree(args.dis_save_path)
    maketree(args.gpt_sample_dir)
    maketree(args.dis_sample_dir)
    maketree(args.gpt_sample_dir2)
    maketree(args.dis_sample_dir2)
    
    maketree(args.log_dir)
    maketree(args.log_path)
    
    
    with open(os.path.join('models', args.model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if args.model_name == '345M':
        args.memory_saving_gradients = True
        if args.optimizer == 'adam':
            args.only_train_transformer_layers = True

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    with tf.Session(config=config) as sess:
        scope_discri='distri'
        
        def get_dis_logit_and_prob_single_step(context, scope):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                context=tf.reshape(context, [-1, args.seq_len])
                emb=tf.get_variable(name='emb', initializer=tf.random.normal([hparams.n_vocab, 32], 0, 0.02))
                context_emb=tf.nn.embedding_lookup(emb, context)
                logit=dis(context_emb, scope=scope_discri)
                prob=tf.sigmoid(logit+1e-7)
            return logit, prob
        
        def get_dis_logit_and_prob(context, context_len, scope):
            ##Pay attention to context_len here. temporary changes!!!!!!!!!!!!!!!!!!!
            context_mask=(1-tf.sequence_mask(context_len-1, args.seq_len-1, dtype=tf.float32))*1e3
            context_mask2=tf.sequence_mask(context_len-1, args.seq_len-1, dtype=tf.float32)
            ones=tf.ones(shape=[tf.shape(context_len)[0], args.seq_len], dtype=tf.int32)*enc.encoder['<|endoftext|>']
            input_tensor_list=[]
            for i in range(1, args.seq_len):
                input_tensor_list.append(tf.concat([context[:, :i+1], ones[:,i+1:]], axis=1))
            input_tensor=tf.concat(input_tensor_list, axis=0)
            log_prob, _=get_dis_logit_and_prob_single_step(input_tensor, scope=scope)
            log_prob=tf.transpose(tf.reshape(log_prob, [args.seq_len-1, -1]))
            log_prob+=tf.cast(context_mask, tf.float32)
            log_prob_min=tf.reduce_min(log_prob, axis=1)
            prob_min=tf.exp(log_prob_min)
            return log_prob_min, prob_min, log_prob
        ##Build discriminator
        
        def build_dis_layer(scope):
            context_pos_discri = tf.placeholder(tf.int32, [None, args.seq_len])
            context_pos_discri_len = tf.placeholder(tf.int32, [None])
            context_neg_discri = tf.placeholder(tf.int32, [None, args.seq_len])
            context_neg_discri_len = tf.placeholder(tf.int32, [None])
            
            label_pos_discri=tf.ones([tf.shape(context_pos_discri_len)[0]], dtype=tf.float32)
            label_neg_discri=tf.zeros([tf.shape(context_neg_discri_len)[0]], dtype=tf.float32)
            logit_pos_discri, prob_pos_discri, mask=get_dis_logit_and_prob(context_pos_discri, context_pos_discri_len, scope=scope)
            logit_neg_discri, _, _=get_dis_logit_and_prob(context_neg_discri, context_neg_discri_len, scope=scope)
        
            loss_pre_pos_discri=tf.nn.sigmoid_cross_entropy_with_logits(labels=label_pos_discri, logits=logit_pos_discri)
            loss_pos_discri=tf.reduce_mean(loss_pre_pos_discri)
            loss_pre_neg_discri=tf.nn.sigmoid_cross_entropy_with_logits(labels=label_neg_discri, logits=logit_neg_discri)
            loss_neg_discri=tf.reduce_mean(loss_pre_neg_discri)
            loss_discri=(loss_pos_discri*args.pos_loss_weight+loss_neg_discri)/(1+args.pos_loss_weight)
        
            train_var_list_discri=[x for x in tf.global_variables() if scope in  x.name]
            train_op_discri=tf.train.AdamOptimizer().minimize(loss_discri, var_list=train_var_list_discri)
            var_list_discri=[x for x in tf.global_variables() if scope in  x.name]
            initializer_discri=tf.variables_initializer(var_list_discri)
            saver_discri=tf.train.Saver(var_list=var_list_discri, max_to_keep=1)
            print('discri: {} build succeed!'.format(scope))
            return context_pos_discri,context_pos_discri_len, context_neg_discri,context_neg_discri_len, loss_pos_discri, loss_neg_discri, loss_discri, train_op_discri, initializer_discri, saver_discri, prob_pos_discri, mask, logit_pos_discri
        
        class dis_class:
            def __init__(self, layer_num=1, scope=scope_discri):
                self.model=[]
                self.dis=np.zeros([layer_num], dtype=np.float32)
                print(layer_num)
                for i in range(layer_num):
                    layer={'scope': scope+str(i)}
                    layer['context_pos_discri'],layer['context_pos_discri_len'], layer['context_neg_discri'],layer['context_neg_discri_len'], layer['loss_pos_discri'], layer['loss_neg_discri'], layer['loss_discri'], layer['train_op_discri'], layer['initializer_discri'], layer['saver_discri'], layer['prob_pos_discri'], layer['mask'], layer['logit_pos_discri'] = build_dis_layer(scope+str(i))
                    self.model.append(layer)
            def prob(self, context, context_len, layer=-1):
                if layer==-1:
                    layer=len(self.model)
                prob_final=tf.ones(tf.shape(context)[0], dtype=tf.float32)
                for i in range(layer):
                    item=self.model[i]
                    scope=item['scope']
                    _, prob, _=get_dis_logit_and_prob(context, context_len, scope=scope)
                    prob_final*=prob
                return prob_final
            def log_prob_step(self, context, layer=-1):
                if layer==-1:
                    layer=len(self.model)
                prob_final=tf.ones(tf.shape(context)[0], dtype=tf.float32)
                log_prob_list=[]
                for i in range(layer):
                    item=self.model[i]
                    scope=item['scope']
                    log_prob, prob=get_dis_logit_and_prob_single_step(context, scope=scope)
                    log_prob_list.append(tf.expand_dims(log_prob, 1))
                log_prob_final=tf.concat(log_prob_list, axis=1)
                return log_prob_final
        
        Dis=dis_class(layer_num=args.layer_num)
        
        context = tf.placeholder(tf.int32, [None, None])
        context_len=tf.placeholder(tf.int32, [None])
        context_mask=tf.sequence_mask(context_len-1, args.seq_len-1, dtype=tf.float32)
        context_in=context
        output = model.model(hparams=hparams, X=context_in)
        loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=context[:, 1:], logits=output['logits'][:, :-1])*context_mask
        
        loss=tf.reduce_sum(loss_tensor, axis=1)/(tf.reduce_sum(context_mask, axis=1)+1e-7)
        loss_sen=tf.reduce_sum(loss)
        loss=tf.reduce_mean(loss)
        
        
        if args.val_every > 0:
            def transform_np(x, lift=args.exponential_param):
                x=x-0.5
                x=x+np.abs(x)
                return lift*x**2
            def transform(x, lift=args.exponential_param):
                x=x-0.5
                x=x+tf.abs(x)
                return lift*x**2
            val_context = tf.placeholder(tf.int32, [args.val_batch_size, args.seq_len])
            val_context_len=tf.placeholder(tf.int32, [args.batch_size])
            NLL_bias=tf.placeholder(tf.float32, [])
            val_context_mask=tf.sequence_mask(val_context_len-1, args.seq_len-1, dtype=tf.float32)
            val_output = model.model(hparams=hparams, X=val_context)
            val_loss_tensor =tf.nn.sparse_softmax_cross_entropy_with_logits(labels=val_context[:, 1:], logits=val_output['logits'][:, :-1])*val_context_mask
            val_context_prob_cut=Dis.prob(val_context, val_context_len)
            val_NLL_cut=tf.log(val_context_prob_cut+1e-7)
            
            val_loss=tf.reduce_sum(val_loss_tensor, axis=1)/(tf.reduce_sum(val_context_mask, axis=1)+1e-7)
            val_loss_cut=(tf.reduce_sum(val_loss_tensor, axis=1)+NLL_bias)/(tf.reduce_sum(val_context_mask, axis=1)+1e-7)-val_NLL_cut/tf.cast(val_context_len, tf.float32)
            
            val_loss_sum=tf.reduce_sum(val_loss_tensor, axis=1)
            val_loss_cut_sum=(tf.reduce_sum(val_loss_tensor, axis=1)+NLL_bias)-val_NLL_cut
            
            val_loss_mean=tf.reduce_mean(val_loss)
            val_loss_cut_mean=tf.reduce_mean(val_loss_cut)
            val_loss_summary = tf.summary.scalar('val_loss', val_loss_mean)


        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=args.seq_len,
            context=context,
            batch_size=args.batch_size,
            temperature=1.0,
            top_k=args.top_k,
            top_p=args.top_p,
            start_token=enc.encoder['<|endoftext|>'])

        start_token=enc.encoder['<|endoftext|>']

        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if args.only_train_transformer_layers else all_vars

        if args.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        elif args.optimizer == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
        else:
            exit('Bad optimizer:', args.optimizer)

        if args.accumulate_gradients > 1:
            if args.memory_saving_gradients:
                exit("Memory saving gradients are not implemented for gradient accumulation yet.")
            opt = AccumulatingOptimizer(
                opt=opt,
                var_list=train_vars)
            opt_reset = opt.reset()
            opt_compute = opt.compute_gradients(loss)
            opt_apply = opt.apply_gradients()
            summary_loss = tf.summary.scalar('loss', opt_apply)
        else:
            if args.memory_saving_gradients:
                opt_grads = memory_saving_gradients.gradients(loss, train_vars)
            else:
                opt_grads = tf.gradients(loss, train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            opt_apply = opt.apply_gradients(opt_grads)
            summary_loss = tf.summary.scalar('loss', loss)

        summary_lr = tf.summary.scalar('learning_rate', args.learning_rate)
        summaries = tf.summary.merge([summary_lr, summary_loss])

        summary_log = tf.summary.FileWriter(
            os.path.join(CHECKPOINT_DIR, args.run_name))

        saver = tf.train.Saver(var_list=all_vars, max_to_keep=1)
        
        sess.run(tf.global_variables_initializer())

        if args.restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(CHECKPOINT_DIR, args.run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join('models', args.model_name))
        elif args.restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', args.model_name))
        else:
            ckpt = tf.train.latest_checkpoint(args.restore_from)
        print('Loading checkpoint', ckpt)
        saver.restore(sess, ckpt)

        print('Loading dataset...')
        data_list, data_len = load_dataset(enc, args.train_data_path, args.seq_len)
        data_sampler = Sampler(data_list, data_len )
        if args.val_every > 0:
            val_data_list, val_data_len = load_dataset(enc, args.eval_data_path, args.seq_len)
        print('dataset has', data_sampler.total_size, 'tokens')
        print('Training...')

        if args.val_every > 0:
            # Sample from validation set once with fixed seed to make
            # it deterministic during training as well as across runs.
            val_data_sampler = Sampler(val_data_list, val_data_len, seed=1)
            val_batches = [val_data_sampler.sample(args.batch_size) for _ in range(args.val_batch_count)]

        counter = 0
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, args.run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')
        
        
        def train_step_discri(layer_id=0, mask_train_epoch=0):
            pos_samples, pos_samples_len=data_sampler.sample(args.batch_size)
            neg_samples=generate_negative_sample(layer_id=layer_id)
            neg_samples_len=get_array_len(neg_samples)
            _, loss=sess.run([Dis.model[layer_id]['train_op_discri'], Dis.model[layer_id]['loss_discri']], feed_dict={Dis.model[layer_id]['context_pos_discri']: pos_samples,Dis.model[layer_id]['context_pos_discri_len']: pos_samples_len, Dis.model[layer_id]['context_neg_discri']: neg_samples, Dis.model[layer_id]['context_neg_discri_len']: neg_samples_len})
            return loss
        
        def generate_negative_samples(layer_id, generate_num=args.batch_size):
            result_list=[]
            generate_num_now=0
            samples_mem=[]
            while generate_num_now<generate_num:
                t=time.time()
                sample_id=generate_negative_sample(layer_id=layer_id)
                samples=[]
                t1=time.time()
                selected_id_list=np.arange(len(sample_id))
                t2=time.time()
                result_list.append(sample_id[selected_id_list])
                generate_num_now+=len(selected_id_list)
            return np.concatenate(result_list, axis=0)[:generate_num]
        
        def get_array_len(sample_array):
            lens=[]
            for item in sample_array:
                for i in range(1, len(item)):
                    if item[i]==enc.encoder['<|endoftext|>']:
                        break
                lens.append(i)
            return np.array(lens).astype(np.int32)
        
        def generate_discri_sample3(layer_id=-1, sample_size=10000, save_path='/mnt/cephfs_new_wj/mlnlp/miaoning/Experiment/gpt-2-sep/samples/discri/sample2.txt'):
            samples=[]
            while len(samples)<sample_size:
                sample_id=generate_negative_sample(layer_id)
                for i in range(len(sample_id)):
                    sample_tem=enc.decode(sample_id[i]).split('<|endoftext|>')[1].split('\n')[0]
                    samples.append(sample_tem)
                print(len(samples))
            with open(save_path, 'w') as g:
                g.write('\n'.join(samples))
        
        
        def eval_discri_NLL(layer_id=0):
            losses_pos=[]
            losses_neg=[]
            for batch in tqdm.tqdm(val_batches):
                pos_samples, pos_samples_len=batch
                neg_samples=generate_negative_sample(layer_id=layer_id)
                neg_samples_len=get_array_len(neg_samples)
                loss_pos, mask=sess.run([Dis.model[layer_id]['loss_pos_discri'], Dis.model[layer_id]['mask']], feed_dict={Dis.model[layer_id]['context_pos_discri']: pos_samples, Dis.model[layer_id]['context_pos_discri_len']: pos_samples_len})
                #print(mask)
                loss_neg=sess.run(Dis.model[layer_id]['loss_neg_discri'], feed_dict={Dis.model[layer_id]['context_neg_discri']: neg_samples, Dis.model[layer_id]['context_neg_discri_len']: neg_samples_len})
                losses_pos.append(loss_pos)
                losses_neg.append(loss_neg)
            return np.mean(losses_pos), np.mean(losses_neg)
        
        def get_discri_quantile(layer_id=0, quantile=0.85):
            logits_list=[]
            for batch in tqdm.tqdm(val_batches):
                pos_samples, pos_samples_len=batch
                logits, mask=sess.run([Dis.model[layer_id]['logit_pos_discri'], Dis.model[layer_id]['mask']], feed_dict={Dis.model[layer_id]['context_pos_discri']: pos_samples, Dis.model[layer_id]['context_pos_discri_len']: pos_samples_len})
                print(np.min(mask, axis=1)[:10])
                print(logits[:10])
                with open('mask.pkl', 'wb') as g:
                    pkl.dump(mask, g)
                logits_list.extend(list(logits))
                break
            with open('logits.pkl', 'wb') as g:
                pkl.dump(sorted(logits_list), g)
            #print(sorted(logits_list))
            print('finish')
            return sorted(logits_list)[int(len(logits_list)*(1-quantile))]
        
        def train_discri(train_step, eval_every, train_layer_list=list(range(len(Dis.model)))):
            #sess.run(initializer_discri)
            print('Start Discri training')
            train_losses=[]
            for layer_id in train_layer_list:
                flag=0
                for epoch in range(train_step):
                    if epoch % eval_every==0:
                        train_losses=np.mean(train_losses)
                        train_losses=[]
                    
                        eval_NLL_pos, eval_NLL_neg=eval_discri_NLL(layer_id)
                        eval_loss=(eval_NLL_pos*args.pos_loss_weight+eval_NLL_neg)/(args.pos_loss_weight+1)
                        print('layer_id:{} discri eval loss:{}'.format(layer_id, eval_loss))
                        print('layer_id:{} discri NLL pos: {}, discri NLL neg: {}'.format(layer_id, eval_NLL_pos, eval_NLL_neg))
                        print(epoch)
                        if epoch==0:
                            eval_loss_old=eval_loss
                        else:
                            print(eval_loss, eval_loss_old)
                            if eval_loss<eval_loss_old:
                                eval_loss_old=eval_loss
                                save_path=args.dis_save_path+str(layer_id)+'/'
                                if not os.path.isdir(save_path):
                                    os.mkdir(save_path)
                                Dis.model[layer_id]['saver_discri'].save(sess, save_path+'a')
                                print('model discri saved!')
                                flag=0
                            else:
                                if epoch>=200:
                                    flag+=1
                            if flag>=4:
                                break
                    train_loss=train_step_discri(layer_id)
                    print('layer_id:{} discri train loss:{}'.format(layer_id, train_loss))
                    train_losses.append(train_loss)
            return eval_loss_old
        
        tf_sample_0 = sample_link.sample_sequence(
                    hparams=hparams,
                    length=args.seq_len,
                    context=context,
                    batch_size=args.batch_size,
                    temperature=1.0,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    start_token=enc.encoder['<|endoftext|>'])
        tf_sample_dict={}
        
        def generate_negative_sample(layer_id=0):
            ##output the filtered result of layer layer_id-1
            if layer_id==0:
                tf_sample=tf_sample_0
                sample = data_sampler.sample(args.batch_size)[0][:,0:1]
                out = sess.run(
                        tf_sample,
                        feed_dict={context: sample})[:,:args.seq_len]
                for i in range(len(out)):
                    flag=0
                    for j in range(len(out[i])):
                        if flag==2:
                            out[i][j]=start_token
                            continue
                        if out[i][j]==start_token:
                            flag+=1
                return out
            else:
                if layer_id==-1:
                    layer_id=len(Dis.model)
                if layer_id in tf_sample_dict:
                    tf_sample=tf_sample_dict[layer_id]
                else:
                    tf_sample = sample_link.sample_sequence_ISMC_threshold(
                        Dis=Dis,
                        layer=layer_id, 
                        hparams=hparams,
                        length=args.seq_len,
                        context=context,
                        batch_size=args.batch_size,
                        temperature=1.0,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        start_token=enc.encoder['<|endoftext|>'])
                    tf_sample_dict[layer_id]=tf_sample
                
                sample = data_sampler.sample(args.batch_size)[0][:,0:1]
                
                out = sess.run(
                        tf_sample,
                        feed_dict={context: sample})[:,:args.seq_len]
                for i in range(len(out)):
                    flag=0
                    for j in range(len(out[i])):
                        if flag==2:
                            out[i][j]=start_token
                            continue
                        if out[i][j]==start_token:
                            flag+=1
                return out

        def validation():
            print('Calculating validation loss...')
            start_time=time.time()
            losses = []
            rates=[]
            for batch in tqdm.tqdm(val_batches):
                losses.append(sess.run(val_loss_mean, feed_dict={val_context: batch[0], val_context_len: batch[1]}))
            v_val_loss = np.mean(losses)
            v_summary = sess.run(val_loss_summary, feed_dict={val_loss_mean: v_val_loss})
            summary_log.add_summary(v_summary, counter)
            summary_log.flush()
            print(
                '[{counter} | {time:2.2f}] validation loss = {loss:2.2f}'
                .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_val_loss))
            return v_val_loss

        def validation_cut(NLL_bias_0=0):
            print('Calculating validation loss...')
            losses = []
            rates=[]
            for batch in tqdm.tqdm(val_batches):
                losses.append(sess.run(val_loss_cut_mean, feed_dict={val_context: batch[0], val_context_len: batch[1], NLL_bias:NLL_bias_0}))
            v_val_loss = np.mean(losses)
            print(
                '[{counter} | {time:2.2f}] validation cut loss = {loss:2.2f}'
                .format(
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_val_loss))
            return v_val_loss

        def sample_batch():
            return [data_sampler.sample(1024) for _ in range(args.batch_size)]
        
        def train_gpt():
            val_loss_old=10000.0
            avg_loss = (0.0, 0.0)
            start_time = time.time()
            counter=0
            while True:
                #pretraining
                if counter % args.save_every == 0:
                    pass
                    #save()
                if counter % args.sample_every == 0:
                    pass
                    #generate_samples()
                if args.val_every > 0 and (counter % args.val_every == 0 or counter == 1):
                    val_loss_1=validation()
                    print(str(counter //args.val_every))
                    if val_loss_1>=val_loss_old:
                        print('pre-training ends!')
                        break
                    else:
                        val_loss_old=val_loss_1
                        saver.save(sess, args.gpt_save_path+'a')
                        print('save succeed!')

                if args.accumulate_gradients > 1:
                    sess.run(opt_reset)
                    for _ in range(args.accumulate_gradients):
                        batch, batch_len=data_sampler.sample(args.batch_size)
                        sess.run(
                            opt_compute, feed_dict={context: batch, context_len:batch_len})
                    (v_loss, v_summary) = sess.run((opt_apply, summaries))
                else:
                    batch, batch_len=data_sampler.sample(args.batch_size)
                    (_, v_loss, v_summary) = sess.run(
                        (opt_apply, loss, summaries),
                        feed_dict={context: batch, context_len:batch_len})

                summary_log.add_summary(v_summary, counter)

                avg_loss = (avg_loss[0] * 0.9 + v_loss,
                            avg_loss[1] * 0.9 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1]))

                counter += 1
        class log_writer:
            def __init__(self, path):
                self.path=path
                with open(path, 'w') as g:
                    g.write('')
            def __call__(self, string, verbose=False):
                with open(self.path, 'a') as g:
                    g.write(string+'\n')
                if verbose:
                    print(string)
        
        try:
            if args.finetune:
                #Finetune GPT-2
                train_gpt() 
            if True:
                #Restore Finetuned model
                save_path=tf.train.latest_checkpoint(args.gpt_save_path)
                saver.restore(sess, save_path)
                print('Load gpt2 succeeded!')
            if args.evaluate_finetune:
                #Evaluate finetuning baseline
                print(validation())
            if args.evaluate_finetune:
                #Calculate reverse-ppl for finetuning baseline
                sample_path=args.gpt_sample_dir2+'sample.txt'
                generate_discri_sample3(layer_id=0, sample_size=3000, save_path=sample_path)
                rev_ppl=train.file_f(train_data_path=sample_path, val_data_path=args.eval_data_path)
                Log_writer=log_writer(args.log_path+'finetune')
                Log_writer('finetuning_rev_ppl: {}'.format(rev_ppl), verbose=True)
            ##Begin tailoring
            if True:
                Log_writer=log_writer(args.log_path+'discri')
                for layer in range(args.layer_num):
                    print(layer)
                    if args.train_tailor:
                        #Train ratio estimator
                        train_discri(500, 10, [layer])
                    if True:
                        #Restore ratio estimator
                        for layer_id in range(layer+1):
                            save_path=args.dis_save_path+str(layer_id)+'/'
                            print(save_path)
                            save_path=tf.train.latest_checkpoint(save_path)
                            print(save_path)
                            Dis.model[layer_id]['saver_discri'].restore(sess, save_path)
                    if False:
                        #Save quantile for analysis
                        with open(args.dis_sample_dir2+'quantile.pkl', 'rb') as f:
                            pkl.load(f)
                        print('Load dis model succeeded!')
                    if True:
                        if layer==0:
                            quantile=0.85
                        else:
                            quantile=0.9
                        Dis.dis[layer]=get_discri_quantile(layer, quantile)
                        with open(args.dis_sample_dir2+'quantile.pkl', 'wb') as g:
                            pkl.dump(Dis.dis, g)
                        print(Dis.dis)
                    if args.evaluate_tailor:
                        #Generate sample for ERS and calculate reverse-ppl
                        sample_path=args.dis_sample_dir2+'_sample_layer_'+str(layer)
                        generate_discri_sample3(layer_id=layer+1, sample_size=3000, save_path=sample_path)
                        rev_ppl=train.file_f(train_data_path=sample_path, val_data_path=args.eval_data_path)
                        Log_writer('layer: {}, dis_rev_ppl: {}'.format(layer, rev_ppl), verbose=True)
        except KeyboardInterrupt:
            print('interrupted')

if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
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




parser = argparse.ArgumentParser(
    description='Fine-tune GPT-2 on your custom dataset.')
#Data config
parser.add_argument('--data_dir', metavar='PATH', type=str, default='./data/', help='dictionary of data, see ./data for examples')
parser.add_argument('--dataset', type=str, default='onto_bn')#parser.add_argument('--dataset', metavar='PATH', type=str, default='/mnt/cephfs_hl/common/lab/miaoning/nmt/workspace2/switchboard/train.src', help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')

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

#Tailor config
parser.add_argument('--exponential_param', type=float, default=2.0)
parser.add_argument('--pos_loss_weight', type=float, default=4.0)

#Path
parser.add_argument('--gpt_save_dir', type=str, default='./models/gen_SMC/')  
parser.add_argument('--dis_save_dir', type=str, default='./models/dis_SMC/')
parser.add_argument('--gpt_sample_dir', type=str, default='./samples/gen_SMC/')
parser.add_argument('--dis_sample_dir', type=str, default='./samples/dis_SMC/')
parser.add_argument('--log_dir', type=str, default='./logs/log_SMC/')

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
    args.dis_save_path=args.dis_save_path+'/'
    maketree(args.dis_save_path)
    maketree(args.gpt_sample_dir2)
    maketree(args.dis_sample_dir2)
    args.dis_sample_dir2=args.dis_sample_dir2+'/'
    maketree(args.dis_sample_dir2)
    
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
        def get_dis_logit_and_prob(context, scope):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                context=tf.reshape(context, [args.batch_size, args.seq_len])
                emb=tf.get_variable(name='emb', initializer=tf.random.normal([hparams.n_vocab, 32], 0, 0.02))
                context_emb=tf.nn.embedding_lookup(emb, context)
                logit=dis(context_emb, scope=scope_discri)
                prob=tf.nn.sigmoid(logit)
            return logit, prob
        ##Build discriminator
        
        def build_dis_layer(scope):
            context_pos_discri = tf.placeholder(tf.int32, [args.batch_size, args.seq_len])
            context_neg_discri = tf.placeholder(tf.int32, [args.batch_size, args.seq_len])
            label_pos_discri=tf.ones([args.batch_size], dtype=tf.float32)
            label_neg_discri=tf.zeros([args.batch_size], dtype=tf.float32)
            logit_pos_discri, prob_pos_discri=get_dis_logit_and_prob(context_pos_discri, scope=scope)
            logit_neg_discri, _=get_dis_logit_and_prob(context_neg_discri, scope=scope)
        
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
            return context_pos_discri, context_neg_discri, loss_pos_discri, loss_neg_discri, loss_discri, train_op_discri, initializer_discri, saver_discri, prob_pos_discri
        
        class dis_class:
            def __init__(self, layer_num=1, scope=scope_discri):
                self.model=[]
                print(layer_num)
                for i in range(layer_num):
                    layer={'scope': scope+str(i)}
                    layer['context_pos_discri'], layer['context_neg_discri'], layer['loss_pos_discri'], layer['loss_neg_discri'], layer['loss_discri'], layer['train_op_discri'], layer['initializer_discri'], layer['saver_discri'], layer['prob_pos_discri'] = build_dis_layer(scope+str(i))
                    self.model.append(layer)
            def prob(self, context, layer=-1):
                if layer==-1:
                    layer=len(self.model)
                prob_final=tf.ones(tf.shape(context)[0], dtype=tf.float32)
                for i in range(layer):
                    item=self.model[i]
                    scope=item['scope']
                    _, prob=get_dis_logit_and_prob(context, scope=scope)
                    prob_final*=prob
                return prob_final
        
        Dis=dis_class(layer_num=1)
        
        context = tf.placeholder(tf.int32, [args.batch_size, None])
        context_len=tf.placeholder(tf.int32, [args.batch_size])
        context_mask=tf.sequence_mask(context_len-1, args.seq_len-1, dtype=tf.float32)
        context_in = randomize(context, hparams, args.noise)
        output = model.model(hparams=hparams, X=context_in)
        loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=context[:, 1:], logits=output['logits'][:, :-1])*context_mask
        
        context_prob_cut=Dis.prob(context)
        
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
            val_context_prob_cut=Dis.prob(val_context)
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
        
        def mask_dis(sample1, sample2):
            for i in range(len(sample1)):
                for j in range(len(sample1[i])):
                    if sample1[i][j]==start_token:
                        l1=j
                        break
                for j in range(len(sample2[i])):
                    if sample2[i][j]==start_token:
                        l2=j
                        break
                l=min(l1,l2)
                if l>=2:
                    l_mask=np.random.randint(0, l, [])
                    for j in range(l_mask, len(sample1[i])):
                        sample1[i][j]=start_token
                    for j in range(l_mask, len(sample2[i])):
                        sample2[i][j]=start_token
            return sample1, sample2
        
        def train_step_discri(layer_id=0, mask_train_epoch=0):
            pos_samples, _=data_sampler.sample(args.batch_size)
            neg_samples=generate_negative_sample_and_diversify(layer_id=layer_id)[:,:args.seq_len]
            _, loss=sess.run([Dis.model[layer_id]['train_op_discri'], Dis.model[layer_id]['loss_discri']], feed_dict={Dis.model[layer_id]['context_pos_discri']: pos_samples, Dis.model[layer_id]['context_neg_discri']: neg_samples})
            for epoch in range(mask_train_epoch):
                pos_samples, _=data_sampler.sample(args.batch_size)
                neg_samples_copy=copy(neg_samples)
                pos_samples, neg_samples_copy=mask_dis(pos_samples, neg_samples_copy)
                _, loss_mask=sess.run([Dis.model[layer_id]['train_op_discri'], Dis.model[layer_id]['loss_discri']], feed_dict={Dis.model[layer_id]['context_pos_discri']: pos_samples, Dis.model[layer_id]['context_neg_discri']: neg_samples_copy})
            return loss
        
        def coverage_rate(list1, list2):
            overlap1=0
            for item in list1:
                if item in list2:
                    overlap1+=1
            overlap2=0
            for item in list2:
                if item in list1:
                    overlap2+=1
            return max((overlap1+0.0)/(len(list1)+0.0), (overlap2+0.0)/(len(list2)+0.0))
        
        def diversify(samples, samples_mem):
            #return list(range(len(samples)))            ##########tem
            sample_split=[]
            for item in samples:
                sample_split.append(item.split())
            sample_selected=[]
            for item in sample_split:
                flag=0
                for item_ref in sample_selected+samples_mem:
                    if coverage_rate(item, item_ref)>0.5:  ##temporary strategy
                        flag=1
                        break
                if flag==0:
                    sample_selected.append(item)
            id_list=[]
            for item in sample_selected:
                id_list.append(sample_split.index(item))
            return id_list
        
        def generate_negative_sample_and_diversify(layer_id, generate_num=args.batch_size):
            result_list=[]
            generate_num_now=0
            samples_mem=[]
            while generate_num_now<generate_num:
                t=time.time()
                sample_id=generate_negative_sample(layer_id=layer_id)[:,:args.seq_len]
                samples=[]
                t1=time.time()
                selected_id_list=np.arange(len(sample_id))
                t2=time.time()
                result_list.append(sample_id[selected_id_list])
                generate_num_now+=len(selected_id_list)
                print(generate_num_now, t1-t, t2-t1)
            return np.concatenate(result_list, axis=0)[:generate_num]
            

        def generate_discri_sample2(layer_id=-1, sample_size=10000, save_path=args.dis_sample_dir2+'sample.txt'):
            samples=[]
            sample_id=generate_negative_sample_and_diversify(layer_id, sample_size)
            for i in range(len(sample_id)):
                samples.append(enc.decode(sample_id[i]).split('<|endoftext|>')[1].split('\n')[0])
            print(len(samples))
            with open(save_path, 'w') as g:
                g.write('\n'.join(samples))
        
        def generate_discri_sample3(layer_id=-1, sample_size=10000, save_path='/mnt/cephfs_new_wj/mlnlp/miaoning/Experiment/gpt-2-sep/samples/discri/sample2.txt'):
            samples=[]
            while len(samples)<sample_size:
                sample_id=generate_negative_sample(layer_id)
                for i in range(len(sample_id)):
                    sample_tem=enc.decode(sample_id[i]).split('<|endoftext|>')[1].split('\n')[0]
                    if len(sample_tem.split())>=3:
                        samples.append(sample_tem)
                print(len(samples))
            with open(save_path, 'w') as g:
                g.write('\n'.join(samples))
        
        def generate_sample2(save_path=args.gpt_sample_dir2+'sample.txt'):
            generate_discri_sample3(layer_id=0, sample_size=3000, save_path=save_path)
        
        def eval_discri_NLL(layer_id=0):
            losses_pos=[]
            losses_neg=[]
            for batch in tqdm.tqdm(val_batches):
                pos_samples, pos_len=batch
                neg_samples=generate_negative_sample(layer_id=layer_id)[:,:args.seq_len]
                loss_pos=sess.run(Dis.model[layer_id]['loss_pos_discri'], feed_dict={Dis.model[layer_id]['context_pos_discri']: pos_samples})
                loss_neg=sess.run(Dis.model[layer_id]['loss_neg_discri'], feed_dict={Dis.model[layer_id]['context_neg_discri']: neg_samples})
                losses_pos.append(loss_pos)
                losses_neg.append(loss_neg)
            return np.mean(losses_pos), np.mean(losses_neg)

        def train_discri(train_step, eval_every):
            #sess.run(initializer_discri)
            train_losses=[]
            for layer_id in range(len(Dis.model)):
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
                            if eval_loss<=eval_loss_old+0.01:
                                if eval_loss<eval_loss_old:
                                    eval_loss_old=eval_loss
                                    save_path=args.dis_save_path+str(layer_id)+'/'
                                    if not os.path.isdir(save_path):
                                        os.mkdir(save_path)
                                    Dis.model[layer_id]['saver_discri'].save(sess, save_path+'a')
                                    print('model discri saved!')
                            else:
                                pass
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
                        feed_dict={context: sample})
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
                    tf_sample = sample_link.sample_sequence_SMC(
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
                        feed_dict={context: sample})
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
            #v_summary = sess.run(val_loss_summary, feed_dict={val_loss_mean: v_val_loss})
            #summary_log.add_summary(v_summary, counter)
            #summary_log.flush()
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
                    #generate_sample2(save_path='./samples/finetune/step/'+str(counter //args.val_every)+'.txt')
                    print(str(counter //args.val_every))
                    if val_loss_1>=val_loss_old:
                        print('pre-training ends!')
                        #break
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

        try:
            if args.finetune:
                train_gpt()
            if True:
                save_path=tf.train.latest_checkpoint(args.gpt_save_path)
                saver.restore(sess, save_path)
                print('Load gpt2 succeeded!')
            if args.evaluate_finetune:
                generate_sample2()
                sample_path=args.gpt_sample_dir2+'sample.txt'
                rev_ppl=train.file_f(train_data_path=sample_path, val_data_path=args.eval_data_path)
                print('Rev_ppl for finetuning:{}'.format(rev_ppl))
            ##Begin cutting
            counter=0
            if args.train_tailor:
                train_discri(500, 10)
            if True:
                for layer_id in range(len(Dis.model)):
                    save_path=args.dis_save_path+str(layer_id)+'/'
                    save_path=args.dis_save_path
                    save_path=tf.train.latest_checkpoint(save_path)
                    Dis.model[layer_id]['saver_discri'].restore(sess, save_path)
                print('Load dis model succeeded!')
            if args.evaluate_tailor:
                generate_discri_sample2()
                sample_path=save_path=args.dis_sample_dir2+'sample.txt'
                rev_ppl=train.file_f(train_data_path=sample_path, val_data_path=args.eval_data_path)
                print('Rev_ppl for Tailor-SMC:{}'.format(rev_ppl))

        except KeyboardInterrupt:
            print('interrupted')


if __name__ == '__main__':
    main()

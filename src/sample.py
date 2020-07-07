import tensorflow as tf

import model

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    with tf.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.where(probs_sums < p, logits_sort, tf.ones_like(logits_sort)*1000) # [batchsize, vocab]
        min_logits = tf.reduce_min(logits_masked, axis=1, keepdims=True) # [batchsize, 1]
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0):
    #if start_token is None:
    #    assert context is not None, 'Specify exactly one of start_token and context!'
    #else:
    #    assert context is None, 'Specify exactly one of start_token and context!'
    context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
            ]

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens


def sample_sequence_for_GAN(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0, ST=False, Gumbel_temperature=3.0):
    context = tf.one_hot(tf.fill([batch_size, 1], start_token), hparams.n_vocab, dtype=tf.float32)
    
    def Gumbel_variable(shape, dtype):
        r=tf.random_uniform(shape=shape, dtype=dtype)
        return -tf.log(-tf.log(r))
        
    def step(hparams, tokens, past=None):
        print('tokens:{}'.format(tokens.shape))
        lm_output = model.model_for_GAN(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
        
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1,:])
        b=tf.get_variable(name='s', initializer=0.1, dtype=tf.float32)
        def body(past, prev, output, output2):
            next_outputs = step(hparams,prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            logits += Gumbel_variable(tf.shape(logits), logits.dtype)
            sample_gumbel=tf.nn.softmax(logits/Gumbel_temperature)
            if ST:
                sample=tf.one_hot(tf.argmax(sample_gumbel, axis=-1), tf.shape(sample_gumbel)[-1], dtype=tf.float32)
            else:
                sample=sample_gumbel
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                sample,
                tf.concat([output, tf.expand_dims(sample, axis=1)], axis=1),
                tf.concat([output2, tf.expand_dims(sample_gumbel, axis=1)], axis=1),
            ]

        def cond(*args):
            return True

        _, _, tokens_out, tokens_gumbel = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1, :],
                context,
                context,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None, None]),
                tf.TensorShape([batch_size, None, None]),
            ],
            back_prop=True,
            swap_memory=True,
        )
        
        return tokens_out, tokens_gumbel

def sample_sequence_SMC(*, Dis, layer, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0):
    #if start_token is None:
    #    assert context is not None, 'Specify exactly one of start_token and context!'
    #else:
    #    assert context is None, 'Specify exactly one of start_token and context!'
    context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])
        shape1=model.past_shape(hparams=hparams, batch_size=batch_size)
        shape1[-2]=-1
        def body(past, prev, output, stop_before):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            
            past=tf.concat([past, next_outputs['presents']], axis=-2)
            prev=tf.squeeze(samples, axis=[1])
            output=tf.concat([output, samples], axis=1)
            output_full=tf.concat([output, tf.zeros([hparams.batch_size, hparams.seq_len+1-tf.shape(output)[1]], dtype=tf.int32)+start_token], axis=1)[:, :hparams.seq_len]
            
            def transform(x, lift=2.0):
                x=x-0.5
                x=x+tf.abs(x)
                return lift*x**2
            prob=transform(Dis.prob(output_full, layer=layer))
            ids=tf.range(hparams.batch_size)
            already_end_ids=ids[:stop_before]
            end_ids=tf.cast(tf.where(tf.equal(prev[stop_before:], start_token))[:,0], dtype=tf.int32)+stop_before
            non_end_ids=tf.cast(tf.where(tf.not_equal(prev[stop_before:], start_token))[:,0], dtype=tf.int32)+stop_before
            
            
            prob_non_end=tf.gather(prob, non_end_ids)
            def true_fn():
                return tf.gather(non_end_ids, tf.random.multinomial(logits=tf.log(prob_non_end+1e-7)[tf.newaxis,:], num_samples=tf.shape(non_end_ids)[0])[0])
            def false_fn():
                return non_end_ids
            sample_ids=tf.cond(tf.shape(non_end_ids)[0]>0, true_fn=true_fn, false_fn=false_fn)
            
            combine_ids=tf.concat([already_end_ids, end_ids, sample_ids], axis=0)
            return [
                tf.reshape(tf.gather(past, combine_ids), shape1),
                tf.reshape(tf.gather(prev, combine_ids), [batch_size]),
                tf.reshape(tf.gather(output, combine_ids), [batch_size, -1]),
                tf.shape(end_ids)[0]+stop_before,
            ]

        def cond(*args):
            return True

        _, _, tokens, _ = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
                0,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([])
            ],
            back_prop=False,
        )

        return tokens

def sample_sequence_ISMC(*, Dis, layer, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0):
    #if start_token is None:
    #    assert context is not None, 'Specify exactly one of start_token and context!'
    #else:
    #    assert context is None, 'Specify exactly one of start_token and context!'
    batch_size=1000
    context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])
        shape1=model.past_shape(hparams=hparams, batch_size=batch_size)
        shape1[-2]=-1
        def body(past, prev, output, stop_before):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            
            past=tf.concat([past, next_outputs['presents']], axis=-2)
            prev=tf.squeeze(samples, axis=[1])
            output=tf.concat([output, samples], axis=1)
            def transform(x, para=2):
                #return x*0+1
                return 1-(1-x)**para
            prob=Dis.prob_step(tf.concat([output, tf.ones(shape=[tf.shape(output)[0], length-tf.shape(output)[1]], dtype=tf.int32)*start_token], axis=1), layer=layer)
            prob=transform(prob)
            ids=tf.range(tf.shape(prob)[0])
            #already_end_ids=ids[:stop_before]
            #end_ids=tf.cast(tf.where(tf.equal(prev[stop_before:], start_token))[:,0], dtype=tf.int32)+stop_before
            #non_end_ids=tf.cast(tf.where(tf.not_equal(prev[stop_before:], start_token))[:,0], dtype=tf.int32)+stop_before
            non_end_ids=ids
            
            prob_non_end=tf.gather(prob, non_end_ids)
            r=tf.random_uniform(shape=tf.shape(prob_non_end)[0:1], dtype=tf.float32)
            selected_ids_pre=tf.where(tf.less(r, prob_non_end))[:,0]
            def true_fn():
                return tf.gather(non_end_ids, selected_ids_pre)
            def false_fn():
                return non_end_ids
            sample_ids=tf.cond(tf.shape(non_end_ids)[0]>0, true_fn=true_fn, false_fn=false_fn)
            #sample_ids=non_end_ids
            
            #combine_ids=tf.concat([already_end_ids, end_ids, sample_ids], axis=0)
            combine_ids=sample_ids
            end_ids=sample_ids
            return [
                tf.gather(past, combine_ids),
                tf.gather(prev, combine_ids),
                tf.gather(output, combine_ids),
                tf.shape(end_ids)[0]+stop_before,
            ]

        def cond(*args):
            return True

        _, _, tokens, _ = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length-1,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
                0,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=None)),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([])
            ],
            back_prop=False,
        )

        return tokens

def sample_sequence_ISMC_threshold(*, Dis, layer, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0):
    #if start_token is None:
    #    assert context is not None, 'Specify exactly one of start_token and context!'
    #else:
    #    assert context is None, 'Specify exactly one of start_token and context!'
    
    batch_size=1000
    context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])
        shape1=model.past_shape(hparams=hparams, batch_size=batch_size)
        shape1[-2]=-1
        def body(past, prev, output, stop_before):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            
            past=tf.concat([past, next_outputs['presents']], axis=-2)
            prev=tf.squeeze(samples, axis=[1])
            output=tf.concat([output, samples], axis=1)

            log_prob=Dis.log_prob_step(tf.concat([output, tf.ones(shape=[tf.shape(output)[0], length-tf.shape(output)[1]], dtype=tf.int32)*start_token], axis=1), layer=layer)
            log_prob_cut=tf.reduce_min(log_prob-Dis.dis[:layer], axis=1)
            ids=tf.range(tf.shape(log_prob_cut)[0])
            #already_end_ids=ids[:stop_before]
            #end_ids=tf.cast(tf.where(tf.equal(prev[stop_before:], start_token))[:,0], dtype=tf.int32)+stop_before
            #non_end_ids=tf.cast(tf.where(tf.not_equal(prev[stop_before:], start_token))[:,0], dtype=tf.int32)+stop_before
            non_end_ids=ids
            
            log_prob_non_end=tf.gather(log_prob_cut, non_end_ids)
            #r=tf.random_uniform(shape=tf.shape(prob_non_end)[0:1], dtype=tf.float32)
            #selected_ids_pre=tf.where(tf.less(r, prob_non_end))[:,0]
            selected_ids_pre=tf.where(tf.less(0.0, log_prob_non_end))[:,0]

            def true_fn():
                return tf.gather(non_end_ids, selected_ids_pre)
            def false_fn():
                return non_end_ids
            sample_ids=tf.cond(tf.shape(non_end_ids)[0]>0, true_fn=true_fn, false_fn=false_fn)
            #sample_ids=non_end_ids
            
            #combine_ids=tf.concat([already_end_ids, end_ids, sample_ids], axis=0)
            combine_ids=sample_ids
            end_ids=sample_ids
            return [
                tf.gather(past, combine_ids),
                tf.gather(prev, combine_ids),
                tf.gather(output, combine_ids),
                tf.shape(end_ids)[0]+stop_before,
            ]

        def cond(*args):
            return True

        _, _, tokens, _ = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length-1,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
                0,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=None)),
                tf.TensorShape([None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([])
            ],
            back_prop=False,
        )

        return tokens
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


def sample_sequence_for_GAN(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0, ST=False, Gumbel_temperature=1.0):
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
        TA=tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=1)
        TA_gumbel=tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=1)
        def body(past, prev, output, output2, step_0):
            step_0=tf.cast(step_0, tf.int32)
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
                TA.write(step_0, sample),
                TA_gumbel.write(step_0, sample_gumbel),
                step_0+1
            ]

        def cond(*args):
            return True

        _, _, TA_tokens_out, TA_tokens_out_gumbel, _ = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1, :],
                TA,
                TA_gumbel,
                tf.zeros([], dtype=tf.int32)
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape(None),
                tf.TensorShape(None),
                tf.TensorShape([])
            ],
            back_prop=True,
        )
        tokens_out=tf.transpose(TA_tokens_out.stack(), [1,0,2])
        tokens_out_gumbel=tf.transpose(TA_tokens_out_gumbel.stack(), [1,0,2])
        return tf.concat([context, tokens_out], axis=1), tf.concat([context, tokens_out_gumbel], axis=1)

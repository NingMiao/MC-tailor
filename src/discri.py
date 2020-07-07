import tensorflow as tf
import numpy as np

def dis(input_sen, scope='dis'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        conv_layer1=tf.layers.conv1d(input_sen, 10, 5)
        conv_layer1=tf.nn.relu(conv_layer1)
        conv_layer2=tf.layers.conv1d(conv_layer1, 5, 5)
        conv_layer2=tf.nn.relu(conv_layer2)
        reshape_layer=tf.reshape(conv_layer2, shape=[tf.shape(conv_layer2)[0], conv_layer2.shape[1]*conv_layer2.shape[2]])
        line_layer1=tf.layers.dense(inputs=reshape_layer, units=10, activation=tf.nn.relu)
        out=tf.layers.dense(line_layer1, 1)
    return tf.squeeze(out, axis=-1)
    
    
if __name__=='__main__':
    A=tf.placeholder(shape=[10,20,30], dtype=tf.float32)
    B=dis(A)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(B, feed_dict={A: np.random.random([10,20,30])}))
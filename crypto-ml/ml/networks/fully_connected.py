'''
Function for developing a fully connected layer
'''

import tensorflow as tf
from .network_helpers import activation_function


def fully_connected(fm_input, output_dims, activation_function, name):

    with tf.variable_scope('fc' + name, reuse=False):
        W = tf.get_variable('W', shape=[fm_input.shape[-1], output_dims])
        b = tf.get_variable('b', shape=[output_dims])
        
        # a = W*X + b, with op names for the graph
        z = tf.matmul(fm_input, W)
        activation = tf.add(z, b)

        return activation_function(activation, activation_function)

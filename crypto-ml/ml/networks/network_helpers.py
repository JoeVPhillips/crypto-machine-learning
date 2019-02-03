import tensorflow as tf


def activation_function(activation, activation_function):
    '''
    Helper function to act as a switch statement for deciding activation function to use in NN layers.
    '''

    if activation_function == 'linear':
        # No activation function used - linear case
        pass
    elif activation_function == 'relu':
        output = tf.nn.relu(activation)
    elif activation_function == 'leaky_relu':
        output = tf.nn.leaky_relu(activation)
    elif activation_function == 'softmax':
        output = tf.nn.softmax(activation)
    else:
        raise KeyError('Invalid activation function provided in fully_connected layer')

    return output
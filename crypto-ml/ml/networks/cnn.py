'''
Convolution Neural Network

Comprised of:
    • Convolutional layers
    • ReLU activation functions
    • Fully connected layers at the end to understand context
    • Softmax activation for output

Reference: https://docs.google.com/document/d/1FurWdmtMsLRuX_UxErxL8MKqbq24oAKJqzceeydk-24/edit
'''

import tensorflow as tf


def cnn_layer(input, filter, strides, padding, activation_function, alpha=0.2):
        '''
        Defines a convolutional layer with activation function.
        Pooling is left to a separate layer as it is not always desired.

        strides = list of ints - stride of sliding window for each dimension of input
        padding = string ("SAME" or "VALID") - type of padding to use
        activation = type of activation to use between CNN layers
        alpha = for leaky_relu - slope of activation function when x<0
        '''

        features = tf.nn.conv2d(input=input, filter=filter, strides=strides, padding=padding)
        return activation_function(features, activation_function)


'''
class CNN:
    # TODO: consider de-classifying

    def __init__(self, activation):
        # TODO: Placeholders etc.

    def cnn_layer(self, input, filter, conv_strides, padding, activation_function, alpha=0.2):
        # TODO: move cnn_layer code in from above?
        
        features = tf.nn.conv2d(input=input, filter=filter, strides=conv_strides, padding=padding)
        return activation_function(features, activation_function)

'''     

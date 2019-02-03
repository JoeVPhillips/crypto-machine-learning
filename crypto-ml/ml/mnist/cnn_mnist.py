'''
CNN to classify handwritten digits for the MNIST datasets
'''

import tensorflow as tf
from networks.cnn import cnn_layer
from networks.fully_connected import fully_connected

# n * (convolutional + ReLU layers) + (FC + ReLU) + (FC + softmax) 


class CNN:

    def __init__(self):
        # TODO: pass a config
        # TODO: THIS MODEL WILL HAVE e.g. def build_train_op etc..
        # WHEN GENERALISING TO OTHER MODELS:
        #   1. EITHER MOVE THESE TO HELPERS AND PASS IN ON INIT.
        #   2. OR MAKE A PARENT CLASS AND SUPER THEM IN.
        #       ENFORCE OVERRIDING OF CUSTOM FUNCTIONS LIKE build_model.
        pass

    def build_model(self, features, n_classes):
        
        with tf.variable_scope('cnn'):

            # Input Layer - reshape as needed - flatten (MNIST)
            with tf.variable_scope('input_transform'):
                input_features = features
                input_features = tf.reshape(input_features, shape=[-1])
                input_features = tf.Print(input_features, [tf.shape(features), tf.shape(input_features)], message='input vs. flattened input')


            # TODO: Generalise this block later using [filter_size1, filter_size2, filter_size3]
            # to give as many layers as in input arg for conv filter sizes - [7, 5, 3] --> 3

            # Hidden layer feature maps
            with tf.variable_scope('filter1'):
                # Define filter and corresponding padding to preserve spatial dimensions of input
                f = 7
                p = (f-1) // 2
                
                # Convolutional + ReLU Layers
                fm1 = cnn_layer(input=input_features, filter=f, strides=(1,1), padding=p, activation_function='relu')
                fm1 = tf.Print(fm1, [tf.shape(fm1)], message='fm1_shape')
            
            with tf.variable_scope('filter2'):
                # Define filter and corresponding padding to preserve spatial dimensions of input
                f = 5
                p = (f-1) // 2

                # Convolutional + ReLU Layers
                fm2 = cnn_layer(input=fm1, filter=f, strides=(1,1), padding=p, activation_function='relu')
                fm2 = tf.Print(fm2, [tf.shape(fm2)], message='fm2_shape')

            with tf.variable_scope('filter3'):
                # Define filter and corresponding padding to preserve spatial dimensions of input
                f = 3
                p = (f-1) // 2

                # Convolutional + ReLU Layers
                fm3 = cnn_layer(input=input_features, filter=f, strides=(1,1), padding=p, activation_function='relu')
                fm3 = tf.Print(fm3, [tf.shape(fm3)], message='fm3_shape')

            # Pooling
            with tf.variable_scope('pooling_layer'):
                pooled_layer = tf.nn.max_pool(value, ksize, strides, padding)
                pooled_layer = tf.Print(pooled_layer, [tf.shape(pooled_layer)], message='pooled_layer')

            # Fully Connected Layer (need to flatten)
            with tf.variable_scope('fc_layer'):
                fc = fully_connected(input=pooled_layer, output_dims=[], activation_function='relu', name='fc')
                fc = tf.Print(fc, [tf.shape(fc)], message='fc_layer')

            with tf.variable_scope('dropout'):
                dropped = tf.nn.dropout(x=fc, keep_rate=0.9)
                dropped = tf.Print(dropped, [tf.shape(dropped)], message='dropped')

            # Output Layer
            with tf.variable_scope('output_layer'):
                # No activation here - this layer outputs logits. Softmax is used as 'activation' to convert to probabilities.
                logits = fully_connected(input=dropped, output_dims=[-1, n_classes], activation_function='linear', name='logits')
                logits = tf.Print(logits, [tf.shape(logits)], message='output_layer')

            probs = tf.nn.softmax(logits)

        return probs, logits
                

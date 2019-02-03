'''
Trainer class to take in input as batches and supply to model
'''


import tensorflow as tf


class Trainer:

    def __init__(self, config, model, sess):
        # Initialise
        features, labels = self.initialise_placeholders(config['n_input_features'], config['n_classes'])
        features = tf.Print(features, [tf.shape(features), tf.shape(labels)], message="features and labels shapes: ")

        # Build model and ops
        # TODO: check features
        '''
        probs, logits = model.build_model(features, labels)

        # TODO: PROBS MIGHT NOT BE ON THE GRAPH - LOGITS SHOULD BE.
        probs = tf.Print(probs, [tf.shape(probs), tf.shape(logits)], message="PROB AND LOGITS SHAPES: ")
        '''

    def initialise_placeholders(self, n_input_features, n_classes):
        '''
        n_input_features = no. of input features e.g. 784 pixels - currently for batch of ONE
        # TODO: change to batch of b.
        n_classes = no. of classes in output
        '''
        features = tf.placeholder(dtype=tf.float32, shape=(None, n_input_features))
        labels = tf.placeholder(dtype=tf.float, shape=(None, n_classes))

        return features, labels
        
    def train(self):
        print("Training")
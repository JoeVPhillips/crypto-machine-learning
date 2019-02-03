'''
Main file for training model
'''

import tensorflow as tf
from mnist.cnn_mnist import CNN
from trainer import Trainer


def main():
    
    # Clear graph stack and reset global graph
    tf.reset_default_graph()

    with tf.Session() as sess:
        # Read data in batches


        # Read in model
        model = CNN()

        # Define inputs
        # TODO: remove hard-coding
        config = {'n_input_features': 784, 'n_classes': 10}

        # Begin training
        print("Training Begin")
        trainer = Trainer(config, model, sess)
        trainer.train()
        print("Training Complete")


if __name__ == '__main__':
        # sys.argv[1]
        main()

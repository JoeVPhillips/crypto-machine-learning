import tensorflow as tf

def mnist():
    '''
    Helper function for loading and preprocessing the ENTIRE MNIST dataset
    '''

    # Load data
    mnist = tf.keras.datasets.mnist

    # Unpack data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalze data between [0,1] --> very. important for performance!
    x_train = x_train/255.0
    x_test = x_test/255.0

    return (x_train, y_train), (x_test, y_test)

# TODO: read in batches!

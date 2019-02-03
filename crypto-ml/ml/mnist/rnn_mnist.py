import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from preprocess_mnist import mnist()

'''
Practice model showing how to set up an RNN using TensorFlow and Keras.

Dense - output layer needs to be dense.
Layer before output layer is also often dense - reconstruction.
'''

(x_train, y_train), (x_test, y_test) = mnist()

# Build sequential model by defining as a sequence of layers
model = Sequential()

# Add an LSTM layer with 128 cells
# Return sequences instead of something flat - would return flat if a dense layer came next.
# If going to another recurrent cell layer, need sequences.
model.add(LSTM(128, input_shape=(x_train.shape[1:]),
                    activation='relu',
                    return_sequences=True))

# Regularise.
model.add(Dropout(0.2))

# Add next layer
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))

# Dense layer
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Output layer - 10 classes (MNIST)
model.add(Dense(10, activation='softmax'))

# Decay - start with larger learning rate for bigger initial steps,
# smaller steps later to not overshoot
optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])  # Metrics to follow

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

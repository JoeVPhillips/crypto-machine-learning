import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time

'''
BatchNormalization normalises between layers
Same reason you'd normalise the inputs to the input layer,
you can normalise the values between each layer (i.e. inputs to each layer).

ModelCheckPoint - can set certain parameters as when you want to save checkpoints.
E.g. incremental stages of validation_accuracy, validation_loss, at max(accuracy), at min(loss), etc..
'''

def run_model(train_x, train_y, validation_x, validation_y, RATIO_TO_PREDICT, SEQ_LEN, FUTURE_PERIOD_PREDICT):

    # EPOCHS = 10
    EPOCHS = 1
    BATCH_SIZE = 64

    # Make name descriptive of the model for tuning in TensorBoard
    NAME = f'{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQS-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}'

    model = Sequential()

    # Add 3 LSTM cell layers. Loops i = 0,1,2.
    # for i in range(3):
    for i in range(1):
        # model.add(LSTM(128, input_shape=(train_x.shape[1:]),
        model.add(LSTM(32, input_shape=(train_x.shape[1:]),
                            activation='relu',
                            return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

    # Reduce dimensionality passed to dense layer if not already handled implicitly
    model.add(Flatten())

    # Add a dense layer
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))

    # (Dense) Output layer. Size is 2 --> 2 classes (buy/not-buy).
    model.add(Dense(2, activation='softmax'))

    # Specify optimizer
    optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6)

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    # Callbacks
    tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

    # Unique file name including epoch and the validation accuracy for that epoch
    filepath = 'RNN_Final-{epoch:02d}-{val_acc:.3f}'

    # Checkpoint object - saves only the best versions of the model.
    # Can use this instead of model.save(...).
    checkpoint = ModelCheckpoint('models/{}.model'.format(filepath),
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max')

    history = model.fit(train_x, train_y,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(validation_x, validation_y),
                        callbacks=[tensorboard, checkpoint])

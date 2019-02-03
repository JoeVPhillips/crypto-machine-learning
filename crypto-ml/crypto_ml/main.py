from model import run_model
from pre_processing import construct_main_df, features_and_labels

'''
Main function of project for predicting crypto currency prices with RNN
Predictions are made for a particular cryptocurrency based on the movements
of that currency AND others.

preprocessing.py - creates features and labels from raw data
helpers.py - contains more generic functions, e.g. balancing data and making
one-hot classification vector embeddings.
'''

def main():
    # Use the last 60 periods to predict future (here 1 period = 1 minute)
    SEQ_LEN = 60

    # How many periods forward to predict
    FUTURE_PERIOD_PREDICT = 3

    # Currency ratio to predict
    RATIO_TO_PREDICT = 'LTC_USD'

    main_df = construct_main_df(SEQ_LEN, FUTURE_PERIOD_PREDICT, RATIO_TO_PREDICT)

    # Time series data - reserve the LAST 5-10% of the historic data for testing.
    # Don't shuffle - see RNN google doc.
    # Use sorted to be sure.
    # .index references the index, .values converts to numpy array
    times = sorted(main_df.index.values)

    # Find time representing last 5%
    test_chunk = times[-int(0.05*len(times))]

    # Separate training and validation data
    # Set validation data as any data in the last 5%
    validation_main_df = main_df[(main_df.index >= test_chunk)]

    # Training data is the other 95%
    main_df = main_df[(main_df.index < test_chunk)]

    train_x, train_y = features_and_labels(main_df, SEQ_LEN)
    validation_x, validation_y = features_and_labels(validation_main_df, SEQ_LEN)

    print(f'train data: {len(train_x)} validation: {len(train_x)}')
    print(f'TRAINING:    Do not buys: {train_y.count(0)} buys: {train_y.count(1)}')
    print(f'VALIDATION:  Do not buys: {validation_y.count(0)} buys: {validation_y.count(1)}')

    run_model(train_x, train_y, validation_x, validation_y, RATIO_TO_PREDICT, SEQ_LEN, FUTURE_PERIOD_PREDICT)

main()

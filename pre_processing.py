'''
Unix time stamp, low, high, open, close, volume of trade.
Close = actual price to use.
'''
import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from collections import deque
from helpers import balance_data, classify

''' 
Preprocess dataframe to extract features and labels
Input:
    df - dataframe from which to extract features
    SEQ_LEN = e.g. 60:  Use the last 60 periods to predict future (here 1 period = 1 minute)
'''
def features_and_labels(df, SEQ_LEN):
    # Drop the future column - only needed it to generate the target.
    # Do not want to leave it in now! It would show the network how to predict the future...
    df = df.drop('future', 1)

    # Scale each column
    for col in df.columns:
        # Don't need to scale or normalise target - it is already done.
        if col != 'target':
            # Normalise using pct_change - BTC is a different price to ETH, etc.,
            # and each volume goes up and down by different amounts.
            # So use pct_change --> only want to see trends in movements wrt. other prices.
            df[col] = df[col].pct_change()
            
            # pct_change will cause at least 1 NaN at the start. Drop.
            df.dropna(inplace=True)

            # Scale the values in each column
            df[col] = preprocessing.scale(df[col].values)
    
    # Drop again in case scaling produces NaN
    df.dropna(inplace=True)

    # Deque - append items to the list, once maxlen is reached, pops oldest. FIFO
    # Provides sequences of price and volume for each currency pair
    sequential_data = []
    prev_minutes = deque(maxlen=SEQ_LEN)

    # df.values: df --> list of lists, won't contain time anymore, but will be in same order
    for i in df.values:

        # Sequence is a sequence of lists
        # Drop i == -1 --> 'target'
        prev_minutes.append([n for n in i[:-1]])
        if len(prev_minutes) == SEQ_LEN:

            # np.array(prev_minutes) = sequence of features
            # i[-1] = current label right now based on the full 60 minutes of data
            # print(np.array(prev_minutes))
            sequential_data.append([np.array(prev_minutes), i[-1]])

    random.shuffle(sequential_data)

    balanced_data = balance_data(sequential_data)

    return balanced_data

''' 
Construct main data frame from raw data with relevant columns added in preprocessing
Inputs: 
    SEQ_LEN = e.g. 60:  Use the last 60 periods to predict future (here 1 period = 1 minute).
    RATIO_TO_PREDICT      = currency ratio to predict, default = 'LTC_USD'.
    FUTURE_PERIOD_PREDICT = how many periods forward to predict.
'''
def construct_main_df(SEQ_LEN, FUTURE_PERIOD_PREDICT, RATIO_TO_PREDICT):
    # For merging dataframes
    main_df = pd.DataFrame()

    # Files to use
    ratios = ['BTC_USD', 'LTC_USD', 'ETH_USD', 'BCH_USD']
    for ratio in ratios:
        # f-strings
        dataset = f'crypto_data/{ratio}.csv'

        # Read df for each sheet
        df = pd.read_csv(dataset, names=['time',
                                        'low',
                                        'high',
                                        'open',
                                        'close',
                                        'volume'])
        
        # Rename df - inplace so don't need to redefine
        df.rename(columns={'close': f'{ratio}_close', 'volume': f'{ratio}_volume'}, inplace=True)

        # Set index as time column, not 0, 1, 2, ...
        df.set_index('time', inplace=True)
        
        # Make the df just the close and volume
        df = df[[f'{ratio}_close', f'{ratio}_volume']]

        # If empty, merge dfs
        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)


    # Make a future column in the df by shifting data down by FUTURE_PERIOD_PREDICT
    main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)

    # Make a new column called target
    # list converts output to a list, assign this to a column
    # map the clasify function, along with its parameters
    # N.B. map(function_to_apply, list_of_inputs) - like looping over the function and applying it to each input
    current_column = main_df[f'{RATIO_TO_PREDICT}_close']
    futures_column = main_df[f'{RATIO_TO_PREDICT}_volume']
    main_df['target'] = list(map(classify, current_column, futures_column))

    # print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future', 'target']].head(10))

    return main_df

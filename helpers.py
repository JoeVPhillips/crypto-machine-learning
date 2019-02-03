import numpy as np
import random

''' Classify signal as buy vs. don't buy '''
def classify(current, future):
    # Buy
    if float(future) > float(current):
        return 1
    # Don't buy
    else:
        return 0


''' Balance buys and sells data to roughly 50-50 '''
def balance_data(sequential_data):
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    # Make equal length
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells

    # Shuffle again so data isn't all buys then all sells
    random.shuffle(sequential_data)

    # Split data out for fitting the model
    X = []
    y = []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y

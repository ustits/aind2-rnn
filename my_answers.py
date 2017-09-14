import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import re


def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    for i in range(0, len(series) - window_size):
        X.append(series[i:i + window_size])
    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model


def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?', '"', '\'', '(', ')', '&', '%']
    for symbol in punctuation:
        text = text.replace(symbol, '')

    abbreviations = [r'\betc\b', r'\bDr\b', r'\bCo\b', r'\B-\B']
    for abbr in abbreviations:
        text = re.sub(abbr, '', text)

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass

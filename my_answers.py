import numpy as np
from keras.layers.core import Activation

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
    replace_with_space = [r'\b--\b', r'\b-\b']
    for replacing in replace_with_space:
        text = re.sub(replacing, ' ', text)

    punctuation = ['!', ',', '.', ':', ';', '?', '"', '\'', '(', ')', '&', '%', '/',
                   '*', '$', 'à', 'â', 'è', 'é', '@', '-']
    for symbol in punctuation:
        text = text.replace(symbol, '')

    abbreviations = [r'\betc\b', r'\bDr\b', r'\bCo\b', r'\B-\B']
    for abbr in abbreviations:
        text = re.sub(abbr, '', text)

    for i in range(0, 10):
        text = text.replace(str(i), '')

    return text


def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])

    return inputs, outputs


# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model

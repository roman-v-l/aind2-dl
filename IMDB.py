# Imports
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt


np.random.seed(42)

NUM_WORDS = 2000

# Loading the data (it's preloaded in Keras)
(x_train, y_train), (x_test, y_test) = imdb.load_data(skip_top=0, num_words=NUM_WORDS)

print(x_train.shape)
print(x_test.shape)

print(x_train[0])
print(y_train[0])

# One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=NUM_WORDS)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(len(x_train[0]))
print(x_train[0])

# One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

# TODO: Build the model architecture

model = Sequential()
model.add(Dense(100, activation='softmax', input_shape=(NUM_WORDS,)))
model.add(Dropout(.2))
#model.add(Dense(10, activation='softmax'))
#model.add(Dropout(.1))
model.add(Dense(2, activation='softmax'))

# TODO: Compile the model using a loss function and an optimizer.

model.compile(loss = 'categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.summary()

# TODO: Run the model. Feel free to experiment with different batch sizes and number of epochs.

model.fit(x_train, y_train, epochs=30, batch_size=100, verbose=1)


train_score = model.evaluate(x_train, y_train, verbose=0)
print("Train Accuracy: ", train_score[1])

test_score = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy: ", test_score[1])


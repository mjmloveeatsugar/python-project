# coding = utf-8

import os
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import models, layers
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_file(file_path):
    with open(file_path, 'r', errors='ignore') as fp:
        content = fp.read()
    return content


def readline_file(file_path):
    with open(file_path, 'r', errors='ignore') as fp:
        content = fp.readlines()
    return content


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.3
EMBEDDING_DIM = 100
texts, lables = [], []
lables_index = {'communication': 0,
                'compensation': 1,
                'construction': 2,
                'instant': 3,
                'safety': 4}

for item in os.listdir('word_div'):
    id_ = lables_index[item[:-4]]
    p = os.path.join('word_div', item)
    content = readline_file(p)
    for value in content:
        value = value.replace('\n', '')
        texts.append(value)
        lables.append(id_)
print('%s texts have been found!' % (len(texts)))
# some word need to add

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('%s unique tokens have been found!' % (len(word_index)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
lables = to_categorical(np.asarray(lables), 5)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', lables.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
lables = lables[indices]
samples = int(VALIDATION_SPLIT * data.shape[0])
x_train, x_test = data[:-samples], data[-samples:]
y_train, y_test = lables[:-samples], lables[-samples:]

model = models.Sequential()
model.add(layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(lables_index), activation='softmax'))
sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['acc'])

history = model.fit(x_train, y_train, validation_split=VALIDATION_SPLIT, nb_epoch=100, batch_size=128)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.subplot(121)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(122)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Loss')
plt.legend()
plt.show()

score_ = model.evaluate(x_test, y_test, verbose=0)
print('test score', score_[0])
print('test accuracy', score_[1])

model.save('model.h5')
print('Saved successfully!')

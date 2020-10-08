# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Activation
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np

# Preprocess data
data = pd.read_csv('fer2013.csv')
labels = pd.read_csv('fer2013new.csv')

orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown',
                    'NF']

n_samples = len(data)
w = 48
h = 48

y = np.array(labels[orig_class_names])
X = np.zeros((n_samples, w, h, 1))
for i in range(n_samples):
    X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))

y_mask = y.argmax(axis=-1)
mask = y_mask < orig_class_names.index('unknown')

X = X[mask]
y = y[mask]

# Using the 10 probabilities
y = y[:, :-2] * 0.1

class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

# Normalize image vectors
X = X / 255.0

# Split Data
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=42)
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.111, random_state=42)

# Data augmentation
shift = 0.1
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    height_shift_range=shift,
    width_shift_range=shift)
datagen.fit(x_train)

# Show images
# it = datagen.flow(x_train, y_train, batch_size=1)
# plt.figure(figsize=(10, 7))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(it.next()[0][0], cmap='gray')
#     # plt.xlabel(class_names[y_train[i]])
# plt.show()

print("X_train shape: " + str(x_train.shape))
print("Y_train shape: " + str(y_train.shape))
print("X_test shape: " + str(x_test.shape))
print("Y_test shape: " + str(y_test.shape))
print("X_val shape: " + str(x_val.shape))
print("Y_val shape: " + str(y_val.shape))


def test_model_y(input_shape=(48, 48, 1), classes=7):
    model = Sequential()

    # 1st stage
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 2nd stage
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Dropout(0.5))

    # 3rd stage
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    # 4th stage
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 5th stage
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))

    model.add(Flatten())

    # Fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(classes, activation='softmax'))

    return model


epochs = 100
batch_size = 64

# Training model from scratch
model = test_model_y(input_shape=(h, w, 1), classes=len(class_names))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                    validation_data=(x_val, y_val), verbose=2)

# # Loading JSON model
# json_file = open('model46.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # Loading weights
# model.load_weights('model46.h5')
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print('Model and weights are loaded and compiled.')

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)

# Plot accuracy graph
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.ylim([0, 1.0])
plt.legend(loc='upper left')
plt.show()

# Plot loss graph
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim([0, 3.5])
plt.legend(loc='upper right')
plt.show()

# Serialize and save model to JSON
test_acc = int(test_acc * 10000)
model_json = model.to_json()
with open('Saved-Models\\model' + str(test_acc) + '.json', 'w') as json_file:
    json_file.write(model_json)
# Serialize and save weights to JSON
model.save_weights('Saved-Models\\model' + str(test_acc) + '.h5')
print('Model and weights are saved in separate files.')

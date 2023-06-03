# Two lines that remove tensorflow GPU logs
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.models import Sequential, model_from_json, Model
from keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, Permute, multiply, Reshape, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn import model_selection
from sklearn.metrics import classification_report
from math import ceil

# Data preprocessing
def preprocess_data():
    data = pd.read_csv('fer2013.csv')
    labels = pd.read_csv('fer2013new.csv')
    orig_class_names = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown','NF']
    n_samples = len(data)
    w = 48
    h = 48  
    y = np.array(labels[orig_class_names])
    X = np.zeros((n_samples, w, h, 1))
    for i in range(n_samples):
        X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))

    return X, y


def clean_data_and_normalize(X, y):
    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear','contempt','unknown', 'NF']

    # Using mask to remove unknown or NF images
    y_mask = y.argmax(axis=-1)
    mask = y_mask < orig_class_names.index('unknown')
    X = X[mask]
    y = y[mask]

    # Convert to probabilities between 0 and 1
    y = y[:, :-2] * 0.1

    # Add contempt to neutral and remove it
    y[:, 0] += y[:, 7]
    y = y[:, :7]

    # Normalize image vectors
    X = X / 255.0

    return X, y


def split_data(X, y):
    test_size = ceil(len(X) * 0.1)

    # Split Data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=test_size, random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test


def data_augmentation(x_train):
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        height_shift_range=shift,
        width_shift_range=shift)
    datagen.fit(x_train)
    return datagen

def show_augmented_images(datagen, x_train, y_train):
    it = datagen.flow(x_train, y_train, batch_size=1)
    plt.figure(figsize=(10, 7))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(it.next()[0][0], cmap='gray')
    plt.show()
    
#  Implementing CBAM attention module
def cbam_block(cbam_feature, ratio=8):
    # Channel attention module (CAM)
    cbam_feature = GlobalAveragePooling2D()(cbam_feature)
    cbam_feature = Reshape((1,1,int(cbam_feature.shape[1])))(cbam_feature)
    cbam_feature = Dense(int(cbam_feature.shape[3])//ratio, activation='relu',kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(cbam_feature)
    cbam_feature = Dense(int(cbam_feature.shape[3]), activation='sigmoid', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')(cbam_feature)
    cbam_feature = multiply([cbam_feature, cbam_feature])
    # Spatial attention module (SAM)
    cbam_feature = Permute((3, 1, 2))(cbam_feature)
    cbam_feature = Conv2D(1, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(cbam_feature)
    cbam_feature = Permute((2, 3, 1))(cbam_feature)
    cbam_feature = multiply([cbam_feature, cbam_feature])

    return cbam_feature

# CNN model
def define_model(input_shape=(48, 48, 1), classes=7):
    num_features = 64
    model = Sequential()

    # 1st stage
    model.add(Conv2D(num_features, kernel_size=(5, 5), input_shape=input_shape, padding='same', activation='relu'))
    model.add(Conv2D(num_features, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 2nd stage
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 3rd stage
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(4 * num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 4th stage
    model.add(Conv2D(8 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(8 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(8 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 5th stage
    model.add(Conv2D(8 * num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(8 * num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(8 * num_features, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Fully connected neural networks
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D):
            cbam = cbam_block(layer.output)
            model.layers[i] = Model(inputs=model.inputs, outputs=[layer.output, cbam])
            model.layers.insert(i + 1, Conv2D(layer.filters, kernel_size=layer.kernel_size, padding='same', kernel_initializer='he_normal'))
            model.layers.insert(i + 2, BatchNormalization())
            model.layers.insert(i + 3, Activation('relu'))
    
    return model


def plot_acc(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1.0])
    plt.legend(loc='upper left')
    plt.savefig('accuracy_vgg.png')

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1.0])
    plt.legend(loc='upper right')
    plt.savefig('loss_vgg.png')

def save_model_and_weights(epochs, model, test_acc):
    # Serialize and save model to JSON
    test_acc = int(test_acc * 10000)
    model_json = model.to_json()
    with open('Saved-Models\\model' + str(test_acc) + '.json', 'w') as json_file:
        json_file.write(model_json)
    # Serialize and save weights to JSON
    model.save_weights('Saved-Models\\model' + str(test_acc) + '.h5')
    print('Model and weights are saved in separate files.')


def load_model_and_weights(model_path, weights_path):
    # Loading JSON model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Loading weights
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print('Model and weights are loaded and compiled.')


def run_model():
    fer_classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    X, y = preprocess_data()
    X, y = clean_data_and_normalize(X, y)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y)
    datagen = data_augmentation(x_train)
    # show_augmented_images(datagen, x_train, y_train)
    epochs = 100
    batch_size = 32

    # Training model from scratch
    model = define_model(input_shape=x_train[0].shape, classes=len(fer_classes))
    plot_model(model, to_file="custom_vgg16.png", show_shapes=True, show_layer_names=True)
    model.summary()
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # callback to provide overfitting
    my_callback = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=1e-10)
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                        steps_per_epoch=len(x_train) // batch_size,
                        validation_data=(x_val, y_val), verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    
    # Predicting the labels
    y_pred = model.predict(x_test) 
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    # Saving the report 
    report = classification_report(y_test, y_pred, target_names=fer_classes, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv("Model_performance_report_vgg_cbam.csv", index=True)
    
    # graph plotting
    plot_acc(history)
    plot_loss(history)
    
    # save the model
    save_model_and_weights(epochs, model, test_acc)

run_model()

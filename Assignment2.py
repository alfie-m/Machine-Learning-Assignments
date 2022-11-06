#import numpy
import numpy as np

#import sklearn
from sklearn import datasets

#import tensorflow and silent all warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

#import keras and neccessary modules
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def main():
    print("********************************************************************************************")
    print("********************************COMP219 Assignment 2****************************************")
    print("********************************************************************************************")
    print("Enter f1 to train deep neural network WITHOUT convolutional layer")
    print("Enter f2 to evaluate deep neural netwrok WITHOUT convolutional layer")
    print("Enter f3 to train deep neural network WITH convolutional layer")
    print("Enter f4 to evaluate deep neural network WITH convolutional layer")
    print("OR enter q to quit")
    select = input("Enter option: ")

    if select == "f1":
        if_f1_selected()

    if select == "f2":
        if_f2_selected()

    elif select == "f3":
        if_f3_selected()

    elif select == "f4":
        if_f4_selected()

    elif select == "q":
        if_q_selected()

    else:
        print("\nInvalid option")
        main()

def if_f1_selected():

    #load digits dataset from sklearn
    digits = datasets.load_digits()
    data, target, target_names, images = digits.data, digits.target, digits.target_names, digits.images
    train_images,train_labels,test_images,test_labels = images[:1000],target[:1000],images[1000:],target[1000:]

    # Training parameters
    batch_size = 128
    n_epochs = 25 #epochs equate to number of "passes through" model when training
    n_classes = 10

    #create sequential model
    model = Sequential()

    #add model layers
    #flatten 8 x 8 matrix to 1 x 64 matrix
    model.add(Flatten(input_shape=(8, 8)))
    #128 neuron layer with relu activation function
    model.add(Dense(256, activation='relu'))
    #output layer with 10 neurons for amount of possible outcomes with softmax as probability distribution
    model.add(Dense(n_classes, activation='softmax'))

    #optimizer
    optimizer = Adam(lr=1e-4)

    #compile above CNN_model using inbuilt 'adam' optimizer
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #train CNN_model with training data and x runs through
    model.fit(train_images, train_labels, epochs=n_epochs)

    #evalaute model
    model.evaluate(test_images, test_labels, verbose=2)

    #save model in HDF5 file
    model.save('model.h5')

    print("********************************************************************************************")
    back_to_menu = input("Enter q to return to the menu: ")
    if back_to_menu == "q":
        main()
    else:
        print("\nINVALID INPUT")
        back_to_menu = input("\n\nEnter q to return to menu: ")
        main()

def if_f2_selected():

    #load digits dataset from sklearn
    digits = datasets.load_digits()
    data, target, target_names, images = digits.data, digits.target, digits.target_names, digits.images
    train_images,train_labels,test_images,test_labels = images[:1000],target[:1000],images[1000:],target[1000:]

    #load saved model
    model = load_model('model.h5')

    #evalaute model
    model.evaluate(test_images, test_labels, verbose=2)

    print("********************************************************************************************")
    back_to_menu = input("Enter q to return to the menu: ")
    if back_to_menu == "q":
        main()
    else:
        print("\nINVALID INPUT")
        back_to_menu = input("\n\nEnter q to return to menu: ")
        main()

def if_f3_selected():

    #set up training data and test data from mnist data set
    (CNN_training_data, CNN_training_labels),(CNN_test_data, CNN_test_labels) = mnist.load_data()
    CNN_training_data  = np.expand_dims(CNN_training_data.astype(np.float32) / 255.0, axis=3)
    CNN_test_data = np.expand_dims(CNN_test_data.astype(np.float32) / 255.0, axis=3)
    CNN_training_labels = to_categorical(CNN_training_labels)
    CNN_test_labels = to_categorical(CNN_test_labels)

    # Training parameters
    CNN_batch_size = 128
    CNN_n_epochs = 5
    CNN_n_classes = 10

    #create sequential CNN model
    CNN_model = Sequential()

    #add convolution layers to cnn model
    CNN_model.add(Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1), padding='same'))
    CNN_model.add(Conv2D(16, 3, activation='relu', padding='same'))
    CNN_model.add(MaxPooling2D(pool_size=2, padding='same'))
    CNN_model.add(Conv2D(32, 3, activation='relu', padding='same'))
    CNN_model.add(Conv2D(32, 3, activation='relu', padding='same'))
    CNN_model.add(MaxPooling2D(pool_size=2, padding='same'))

    #add dense layers to CNN model
    CNN_model.add(Flatten())
    CNN_model.add(Dense(128, activation='relu'))
    CNN_model.add(Dropout(0.2))
    CNN_model.add(Dense(CNN_n_classes, activation='softmax'))

    #optimizer
    CNN_optimizer = Adam(lr=1e-4)

    #compile above CNN_model using inbuilt 'adam' optimizer
    CNN_model.compile(loss='categorical_crossentropy', optimizer=CNN_optimizer, metrics=['categorical_accuracy'])

    # Train CNN_model with training data and 5 runss through
    CNN_model.fit(CNN_training_data, CNN_training_labels, epochs=CNN_n_epochs, batch_size=CNN_batch_size, validation_data=(CNN_test_data, CNN_test_labels))

    #evaluate CNN model
    CNN_model.evaluate(CNN_test_data, CNN_test_labels)

    # Save CNN_model in HDF5 file
    CNN_model.save('CNN_model.h5')

    print("********************************************************************************************")
    back_to_menu = input("Enter q to return to the menu: ")
    if back_to_menu == "q":
        main()
    else:
        print("\nINVALID INPUT")
        back_to_menu = input("\n\nEnter q to return to menu: ")
        main()

def if_f4_selected():

    #set up training data and test data from mnist data set
    (CNN_training_data, CNN_training_labels),(CNN_test_data, CNN_test_labels) = mnist.load_data()
    CNN_training_data  = np.expand_dims(CNN_training_data.astype(np.float32) / 255.0, axis=3)
    CNN_test_data = np.expand_dims(CNN_test_data.astype(np.float32) / 255.0, axis=3)
    CNN_training_labels = to_categorical(CNN_training_labels)
    CNN_test_labels = to_categorical(CNN_test_labels)

    #load saved model
    CNN_model = load_model('CNN_model.h5')

    #evaluate CNN model
    CNN_model.evaluate(CNN_test_data, CNN_test_labels)

    print("********************************************************************************************")
    back_to_menu = input("Enter q to return to the menu: ")
    if back_to_menu == "q":
        main()
    else:
        print("\nINVALID INPUT")
        back_to_menu = input("\n\nEnter q to return to menu: ")
        main()

def if_q_selected():
    exit()

main()

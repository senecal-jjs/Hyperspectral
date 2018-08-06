import keras
from keras.models import Sequential
from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import class_weight
from sklearn.metrics import recall_score, precision_score
import numpy as np
import tensorflow as tf
import os, csv, datetime, collections
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CNN:

    def __init__(self, data, wavelengths):
        self.data = np.asarray([np.ndarray.take(np.asarray(x[0]),wavelengths, axis=2) for x in data if x[0] != []])
        self.img_shape = self.data[0].shape
        self.data_shape = ((len(data),), self.img_shape)
        self.data_shape = sum(self.data_shape, ())
        self.labels = np.asarray([x[-1] for x in data if x[0] != []])
        self.unique_labels = set(self.labels)
        self.wavestring = "_".join(str(wave) for wave in wavelengths)

    def train_cnn(self, generator=True):
        input_size = self.img_shape
        print("shape: ", input_size)
        output_size = len(self.unique_labels)

        x_min = self.data.min(axis=(1, 2), keepdims=True)
        x_max = self.data.max(axis=(1, 2), keepdims=True)
        features = (self.data - x_min)/(x_max-x_min)
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(self.labels)
        labels = keras.utils.to_categorical(labels_encoded, output_size)

        """
        Tuning parameters
        """
        pooling1 = "max"
        pooling2 = "max"
        hidden_nodes= 150
        activation_function = "relu"
        filters1 = 40
        filters2 = 30
        kernel1=(7,7)
        kernel2=(5,5)
        pool1=(3,3)
        pool2=(2,2)
        weighted="False"

        """
        10-Fold Cross Validation
        """
        kfold = KFold(n_splits=10, shuffle=True, random_state=5)
        scores = []
        fold=1
        for train, test in kfold.split(features, labels):
          print("fold: ", fold)
          """
          Create Convolutional Network Architecture
          """
          model = Sequential()
          model.add(Conv2D(filters1, kernel_size=kernel1, strides=(2, 2),
                           activation='relu',
                           input_shape=input_size, use_bias=True))
          if pooling1 == "max":
            model.add(MaxPooling2D(pool_size=pool1, strides=(2, 2)))
          elif pooling1 == "avg":
            model.add(AveragePooling2D(pool_size=pool1, strides=(2, 2)))
          model.add(Conv2D(filters2, kernel2, activation= activation_function ))
          if pooling2 == "max":
            model.add(MaxPooling2D(pool_size=pool2))
          elif pooling2 == "avg":
            model.add(AveragePooling2D(pool_size=pool2))
          #model.add(Conv2D(16, kernel2, activation=activation_function))
          #model.add(AveragePooling2D(pool_size=pool2))
          model.add(Flatten())
          model.add(Dense(hidden_nodes, activation= activation_function ))
          #model.add(Dense(hidden_nodes, activation= activation_function ))
          model.add(Dense(output_size, activation='softmax'))

          model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])

          """
          Calculate class weights to potentially even out imbalances
          """
          class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(labels_encoded[train]),
                                                 labels_encoded)

          """
          Fit model with data generator or without
          """
          if generator:
            print("Using data generator.\n")
            datagenerator = image.ImageDataGenerator(featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
            datagenerator.fit(features[train])
            model.fit_generator(datagenerator.flow(features[train], labels[train], batch_size=32), 
              #class_weight=class_weights, 
              steps_per_epoch=len(features[train]) / 32, 
              epochs=3, 
              verbose=2
              )
          else:
            model.fit(features[train], labels[train],
                    validation_split=0.1,
                    batch_size=20,
                    epochs=50,
                    verbose=2,
                    #class_weight=class_weights,
                    validation_data=(features[test], labels[test])
                    #callbacks=callbacks_list
                    )
          """
          Calculate loss, accuracy, precision and recall using MICRO averaging to get best scores based on imbalances
          """
          score = model.evaluate(features[test], labels[test], verbose=2)
          predictions = np.rint(model.predict(features[test]))
          precision = precision_score(labels[test], predictions, average='micro')
          recall = recall_score(labels[test], predictions,  average='micro')
          score = np.append(np.append(score, precision), recall)
          print(score)
          scores.append(score)
          fold = fold+1

        """
        Writing results
        """
        result = np.asarray(scores)
        averages = [np.average(result, axis=0)]
        print("10-fold average:", averages)
        result = np.append(result, averages, axis=0)
        np.savetxt('Results/results_' + self.wavestring + "_" + datetime.datetime.now().strftime("%Y%m%d_%H:%M") +'.csv',result, delimiter=",")
        file = open("Results/results_" + self.wavestring + "_" + datetime.datetime.now().strftime("%Y%m%d_%H:%M") +'_configuration.txt', "w")
        file.write("Data shape: " + str(self.img_shape) + "\nPooling type: " + pooling1 + " - " + pooling2 
          + "\nHidden nodes: " + str(hidden_nodes) + "\nWeighted: " + weighted 
          + "\nActivation: " + activation_function
          + "\nFilters 1: " + str(filters1) + " - Kernel 1: " + str(kernel1) + " - Pooling 1: " + str(pool1) 
          + "\nFilters 2: " + str(filters2) + " - Kernel 2: " + str(kernel2) + " - Pooling 2: " + str(pool2))
    



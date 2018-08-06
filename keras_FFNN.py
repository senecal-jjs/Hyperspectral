import keras
from keras.models import Sequential
from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from keras.layers import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import recall_score, precision_score
import numpy as np
import tensorflow as tf
import os, csv, datetime, collections
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class FFNN:
	def __init__(self, data, wavelengths):
		self.data = np.asarray([np.ndarray.take(np.asarray(x[0]),wavelengths, axis=2) for x in data if x[0] != []])
		self.labels = np.asarray([x[-1] for x in data if x[0] != []])
		self.unique_labels = set(self.labels)
		self.input= np.asarray([np.ndarray.flatten(x) for x in self.data])
		print(self.input)

	def train_ffnn(self):
		output_size = len(self.unique_labels)
		input_size = self.input[0].shape
		scaler = MinMaxScaler()
		features = scaler.fit_transform(self.input)

		encoder = LabelEncoder()
		labels_encoded = encoder.fit_transform(self.labels)
		labels = keras.utils.to_categorical(labels_encoded, output_size)

		hidden_nodes = 250
		activation_function = "relu"

		"""
		10-Fold Cross Validation
		"""
		kfold = KFold(n_splits=10, shuffle=True, random_state=5)
		scores = []
		fold=1
		for train, test in kfold.split(features, labels):
			print("fold: ", fold)
			model = Sequential()
			model.add(Dense(hidden_nodes, input_shape=input_size))
			model.add(Dense(hidden_nodes, activation= activation_function ))
			model.add(Dense(output_size, activation='softmax'))
			model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adam(),
                        metrics=['accuracy'])

			model.fit(features[train], labels[train],
                    validation_split=0.1,
                    batch_size=20,
                    epochs=10,
                    verbose=2,
                    validation_data=(features[test], labels[test]))
			score = model.evaluate(features[test], labels[test], verbose=2)
			print(score)
			print(score)
			scores.append(score)
			fold = fold+1

		result = np.asarray(scores)
		averages = [np.average(result, axis=0)]
		print("10-fold average:", averages)
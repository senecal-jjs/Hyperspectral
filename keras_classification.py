import pickle
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from random import shuffle
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
import utils
import image
import pickle
import os
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt


class classifier:

    def __init__(self, data, norm=True):
        self.data = data
        shuffle(self.data)
        self.inputs = self._extract_input(norm)
        self.labels = self._extract_labels()
        self.valid_labels = self._get_valid_labels()
        self.encoder = LabelEncoder()

    def _extract_input(self, norm):
        """
        Extract the inputs and normalize
        them if normalize is set to true
        """

        inputs = [x[0] for x in self.data]

        if norm:
            inputs = normalize(inputs)

        return inputs

    def _extract_labels(self):
        """
        Extract the labels from the data
        """

        labels = [x[1] for x in self.data]
        return labels

    def _get_valid_labels(self):
        """
        Extract the set of valid class labels
        from the data
        """

        label_list = [x[1] for x in self.data]
        return set(label_list)

    def _ohe_labels(self):
        """
        One hot encode the labels
        """

        self.encoder.fit(self.labels)
        self.labels = self.encoder.transform(self.labels)
        self.labels = np_utils.to_categorical(self.labels)


    def train_mlp(self, cv=False):
        """
        Train the mlp on the training data
        """

        self._ohe_labels()

        input_size = len(self.inputs[0])
        output_size = len(self.valid_labels)
        hidden_size_1 = 15
        hidden_size_2 = 15

        # Create the MLP
        model = Sequential()
        model.add(Dense(hidden_size_1, activation='relu', input_dim=input_size))
        model.add(Dense(hidden_size_2, activation='relu'))
        model.add(Dense(output_size, activation='softmax'))

        # Compile model with optimizer and loss function
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Configure data
        features = self.inputs
        labels = self.labels
        kfold = KFold(n_splits=10, shuffle=True, random_state=5)

        scores = []

        if cv:
            for train, test in kfold.split(features, labels):

                # Create the MLP
                model = Sequential()
                model.add(Dense(hidden_size_1, activation='relu', input_dim=input_size))
                model.add(Dense(hidden_size_2, activation='relu'))
                model.add(Dense(output_size, activation='softmax'))

                # Compile model with optimizer and loss function
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
                model.fit(features[train], labels[train], epochs=25, batch_size=5, verbose=2)

                # evaluate the model
                score = model.evaluate(features[test], labels[test], verbose=2)
                print("{0}: {1}".format(model.metrics_names[1], score[1]))
                scores.append(score[1])

            return scores 

        else:
            model.fit(features, labels, epochs=30, batch_size=5, verbose=2)
            return model



    def classify_new_image(self):
        """
        Train the network, then classify each
        square of a divided image
        """

        n=5
        file_path = 'Data/YukonGold_Tomato_Banana_1_Day3.bil'

        print ("Loading image...")
        raw_image = image.HyperCube(file_path)
        #raw_image.dark_correction()
        original_shape = raw_image.image.shape
        orig_x = original_shape[0]
        orig_y = original_shape[1]

        print ("Dividing image...")
        divided_image_reflectances = utils.avg_spectra_divided_image(raw_image, n)

        input_size = len(self.inputs[0])
        output_size = len(self.valid_labels)
        hidden_size_1 = 15
        hidden_size_2 = 15

        print ("Training model...")
        model = self.train_mlp()

        print ("Classifying image...")
        classified_image = model.predict(divided_image_reflectances)

        number_labels = []
        for im in classified_image:
            number_labels.append(np.argmax(im))

        number_labels = np.array(number_labels).astype(int)

        labeled_image = self.encoder.inverse_transform(number_labels)
        labeled_image = np.reshape(labeled_image, (orig_x/n, orig_y/n, 1))

        raw_image.fix_image()
        divided_image_reflectances = utils.avg_spectra_divided_image(raw_image, n)
        divided_image_reflectances = np.reshape(divided_image_reflectances, (orig_x/n, orig_y/n, 290))
        plt.plot(divided_image_reflectances[0][0])
        plt.show()

        return (labeled_image, divided_image_reflectances)

if __name__ == '__main__':

    try:
        data = pickle.load( open( "avg_class_refl.p", "rb" ))

    except (OSError, IOError) as e:
        "No file found..."

    classify = classifier(data, norm=True)
    #errors = classify.train_mlp(cv=False)
    #for error in errors:
    #    print (error)

    image_classes = classify.classify_new_image()

    pickle.dump(image_classes[0], open("image_labels.p", "wb" ) )
    pickle.dump(image_classes[1], open("image_refl.p","wb"))
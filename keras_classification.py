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
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


class classifier:

    def __init__(self, data, norm=True):
        self.data = data
        shuffle(self.data)
        self.inputs = self._extract_input(norm)
        self.labels = self._extract_labels()
        self.valid_labels = self._get_valid_labels()

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

        encoder = LabelEncoder()
        encoder.fit(self.labels)
        self.labels = encoder.transform(self.labels)
        self.labels = np_utils.to_categorical(self.labels)


    def train_mlp(self, cv=False):
        """
        Train the mlp on the training data
        """

        #self._create_encoder()
        #self._labels_to_floats()
        self._ohe_labels()

        learning_rate = 0.01
        iterations = 2000
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
        #data = preprocess(datafile)
        features = self.inputs
        labels = self.labels
        kfold = KFold(n_splits=10, shuffle=True, random_state=5)

        scores = []

        if cv:
            for train, test in kfold.split(features, labels):
                model.fit(features[train], labels[train], epochs=50, batch_size=5, verbose=2)

                # evaluate the model
                score = model.evaluate(features[test], labels[test], verbose=2)
                print("{0}: {1}".format(model.metrics_names[1], score[1]))
                scores.append(score[1])

            return scores 

        else:
            model.fit(features, labels, epochs=25, batch_size=5, verbose=2)
            return model



    def classify_new_image(self):
        """
        Train the network, then classify each
        square of a divided image
        """

        n=10
        file_path = 'YukonGold_Tomato_Banana_1_Day3.bil'
        print ("Loading image...")
        raw_image = image.HyperCube(file_path)
        raw_image.dark_correction()
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
        
        #classified_image = np.reshape(classified_image, (orig_x/n, orig_y/n, 1))
        #return classified_image

if __name__ == '__main__':

    try:
        data = pickle.load( open( "avg_class_refl.p", "rb" ))

    except (OSError, IOError) as e:
        "No file found..."

    classify = classifier(data, norm=True)
    #classify.train_mlp(cv=True)
    classify.classify_new_image()


    #classify.test()
    #classify.leave_one_out()

    #print float(classify.correct)/len(classify.test_in)

    #image_classes = classify.classify_new_image()

    #pickle.dump(image_classes, open("image_labels.p", "wb" ) )
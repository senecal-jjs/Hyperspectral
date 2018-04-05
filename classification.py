import pickle
import numpy as np
import tensorflow
from sklearn.svm import SVC
from random import shuffle
from sklearn.preprocessing import normalize


class classifier:

    def __init__(self, data, norm=True):
        self.data = data
        shuffle(self.data)
        self.inputs = self._extract_input(norm)
        self.labels = self._extract_labels()
        self.valid_labels = self._get_valid_labels()
        self.train_in, self.train_labels, self.test_in, self.test_labels = self._split_data()
        self.clf = None
        self.classifier_type = None

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

    def _split_data(self):
        """
        Split data into train and test
        """

        #Two thirds train, one third test
        split = 2*(len(self.inputs)/3)
        train_in = self.inputs[:split]
        train_labels = self.labels[:split]
        test_in = self.inputs[split:]
        test_labels = self.labels[split:]

        return train_in, train_labels, test_in, test_labels

    def set_classifier(self, clf_type="svm"):
        """
        Create the specified classifier
        """

        if clf_type == "svm":
            self._create_svm()
        elif clf_type == "knn":
            self._create_knn()
        elif clf_type == "mlp":
            self._create_mlp()
        else:
            print "Invalid classifier selected."

    def _create_svm(self):
        """
        Create an SVM and fit the data
        """

        self.classifier_type = "svm"
        self.clf = SVC()
        self.clf.fit(self.train_in, self.train_labels)

    def _create_knn(self):
        """
        Create knn model with which to 
        classify the data
        """

        self.classifier_type = "knn"

    def _create_mlp(self):
        """
        Create mlp network with which to
        classify the data
        """

        self.classifier_type = "mlp"


    def test(self):
        """
        Run the classifier on the test set
        """

        if self.classifier_type == "svm":
            self._test_svm()

        elif self.classifier_type == "knn":
            self._test_knn()

        elif self.classifier_type == "mlp":
            self._test_mlp()

        else:
            print "Invalid classifier type."

    def _test_svm(self):
        """
        Test the SVM's performance on the data
        """

        for i, test_input in enumerate(self.test_in):

            prediction = self.clf.predict(test_input.reshape(1,-1))
            print prediction, self.test_labels[i]

    def _test_knn(self):
        """
        Test knn's performance on the data
        """

        pass

    def _test_mlp(self):
        """
        Test the MLP's performance on the data
        """

        pass


if __name__ == '__main__':

    try:
        data = pickle.load( open( "avg_whole_refl.p", "rb" ))

    except (OSError, IOError) as e:
        "No file found..."

    classify = classifier(data, norm=True)
    classify.set_classifier('svm')
    classify.test()

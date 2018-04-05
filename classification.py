import pickle
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from random import shuffle
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier


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
        self.clf = KNeighborsClassifier(n_neighbors=1)
        self.clf.fit(self.train_in, self.train_labels)

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

        for i, test_input in enumerate(self.test_in):

            prediction = self.clf.predict(test_input.reshape(1,-1))
            print prediction, self.test_labels[i]

    def _labels_to_floats(self):
        """
        Convert string labels to floats
        """
        for i in range(len(self.train_labels)):
            if str(self.train_labels[i]) == "banana":
                self.train_labels[i] = np.array([1.0, 0.0, 0.0])
            elif str(self.train_labels[i]) == "tomato":
                self.train_labels[i] = np.array([0.0, 1.0, 0.0])
            elif str(self.train_labels[i]) == "potato":
                self.train_labels[i] = np.array([0.0, 0.0, 1.0])
            else:
                print "bad data"


    def _test_mlp(self):
        """
        Test the MLP's performance on the data
        """

        self._labels_to_floats()

        learning_rate = 0.001
        momentum = 0.3
        iterations = 200
        batch_size = 100
        display_step = 1
        input_size = len(self.inputs[0])
        output_size = len(self.valid_labels)
        hidden_size_1 = 15
        hidden_size_2 = 15

        #network weights
        w1 = tf.Variable(tf.random_normal([input_size,hidden_size_1]))
        w2 = tf.Variable(tf.random_normal([hidden_size_1,hidden_size_2]))
        w_out = tf.Variable(tf.random_normal([hidden_size_2,output_size]))

        #layer biases
        b1 = tf.Variable(tf.random_normal([hidden_size_1]))
        b2 = tf.Variable(tf.random_normal([hidden_size_2]))
        b_out = tf.Variable(tf.random_normal([1]))

        #input and output placeholders
        x = tf.placeholder("float", [None, input_size])
        y = tf.placeholder("float", [None, output_size])

        #network layers
        layer1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2))
        layer_out = tf.add(tf.matmul(layer2, w_out), b_out)

        #RMSE loss and momentum optimizer
        #loss = tf.sqrt(tf.reduce_mean(tf.square(layer_out-y)))
        #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
        #                momentum=momentum, use_nesterov=True).minimize(loss)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=layer_out, labels=y))

        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            for i in range(iterations):

                _, l = sess.run([optimizer, loss], feed_dict=
                    {x: self.train_in, y:self.train_labels})
                print(l)

            print("-----------------------------")


            out = sess.run(layer_out, feed_dict={x: self.test_in})
            for i in range(len(self.test_in)):
                #print ("in: " + str(data_in[i]) + "      out: " + str(out[i]) + "   true: " + str(data_out[i]))
                print ("out: " + str(out[i]) + 
                    "   true: " + str(self.test_labels[i]))

if __name__ == '__main__':

    try:
        data = pickle.load( open( "avg_whole_refl.p", "rb" ))

    except (OSError, IOError) as e:
        "No file found..."

    classify = classifier(data, norm=True)
    classify.set_classifier('mlp')
    classify.test()

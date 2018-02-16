from image import HyperCube
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class autoencoder():

    def __init__(self, data, learning_rate = 0.01, iterations = 1000, hidden_layers = [10],
        output_size = None, target_outputs = None, batch_size = 256):

        self.data = data
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.hidden_layers = hidden_layers

        if len(data) != 0:
            self.input_size = len(data[0])
        else:
            print "WARNING: No data found when trying to construct autoencoder"
            self.input_size = -1

        self.output_size = output_size or self.input_size
        self.target_outputs = target_outputs or data
        self.encoder_weights = self.create_weights("encoder")
        self.decoder_weights = self.create_weights("decoder")
        self.encoder_biases = self.create_biases("encoder")
        self.decoder_biases = self.create_biases("decoder")
        self.batch_size = batch_size
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

    def create_weights(self, weight_type):
        """
        Given the input size, output size, number of hidden
        layers, and nodes per hidden layer, create a list
        of the weights between each layer, initialized randomly.
        """

        weights = []

        if self.input_size == -1:
            return None

        elif weight_type == "encoder":
            weights.append(tf.Variable(tf.random_normal([self.input_size, self.hidden_layers[0]])))

            for i in range(len(self.hidden_layers) - 1):
                weights.append(
                    tf.Variable(tf.random_normal(
                        [self.hidden_layers[i], self.hidden_layers[i+1]])))

        elif weight_type == "decoder":
            for i in reversed(range(len(self.hidden_layers) - 1)):
                weights.append(
                    tf.Variable(tf.random_normal(
                        [self.hidden_layers[i+1], self.hidden_layers[i]])))

            weights.append(tf.Variable(
                tf.random_normal([self.hidden_layers[0], self.output_size])))

        else:
            #Should not hit this else statement
            return None

        return weights

    def create_biases(self, bias_type):
        """
        Given the size of each hidden layer, create biases for 
        the network.
        """

        biases = []

        if bias_type == "encoder":
            for layer_size in self.hidden_layers:
                biases.append(tf.Variable(tf.random_normal([layer_size])))

        elif bias_type == "decoder":
            for i in reversed(range(len(self.hidden_layers) - 1)):
                biases.append(tf.Variable(tf.random_normal([self.hidden_layers[i]])))
            biases.append(tf.Variable(tf.random_normal([self.output_size])))

        else:
            #Should not hit this statement
            return None

        return biases

    def create_encoder(self):
        """
        Create sigmoidal encoder based on the weights and biases.
        The encoder does the actual compression of the data.
        """

        layers = []
        data_in = tf.placeholder("float", shape=(1, self.input_size))

        for i, weights in enumerate(self.encoder_weights):
            if i == 0:
                layers.append(tf.nn.sigmoid(tf.add(tf.matmul(data_in, weights),
                                   self.encoder_biases[i])))
            else:
                layers.append(tf.nn.sigmoid(tf.add(tf.matmul(layers[i-1], weights),
                                   self.encoder_biases[i])))

        return layers[-1]

    def create_decoder(self):
        """
        Create sigmoidal decoder based on the weights and biases.
        The decoder does the recovery of the data. The last layer
        defaults to a linear activation function.
        """

        layers = []
        data_in = self.encoder

        for i, weights in enumerate(self.decoder_weights):
            if i == 0:
                layers.append(tf.nn.sigmoid(tf.add(tf.matmul(data_in, weights),
                                   self.decoder_biases[i])))

            elif i == len(self.decoder_weights) - 1:
                #last layer, use linear activation function
                layers.append(tf.nn.relu(tf.add(tf.matmul(layers[i-1], weights),
                                   self.decoder_biases[i])))

            else:
                layers.append(tf.nn.sigmoid(tf.add(tf.matmul(layers[i-1], weights),
                                   self.decoder_biases[i])))

        return layers[-1]


if __name__ == '__main__':

    data = [np.array([1,2,3,5]), np.array([4,2,9,2]), np.array([1.2,7.2,3,5])]
    thing = autoencoder(data, hidden_layers = [3,2])

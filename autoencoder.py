from image import HyperCube
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class autoencoder():

    def __init__(self, data, learning_rate = 0.01, iterations = 1000, hidden_layers = [10],
        output_size = None, target_outputs = None):

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

    def create_weights(self, weight_type):
        """
        Given the input size, output size, number of hidden
        layers, and nodes per hidden layer, create a list
        of the weights between each layer, initialized randomly.
        """

        if self.input_size == -1:
            return None

        elif weight_type == "encoder":
            weights = [tf.Variable(tf.random_normal([self.input_size, self.hidden_layers[0]]))]

            for i in range(len(self.hidden_layers) - 1):
                weights.append(
                    tf.Variable(tf.random_normal(
                        [self.hidden_layers[i], self.hidden_layers[i+1]])))

            return weights

        elif weight_type == "decoder":
            weights = []
            for i in reversed(range(len(self.hidden_layers) - 1)):
                weights.append(
                    tf.Variable(tf.random_normal(
                        [self.hidden_layers[i+1], self.hidden_layers[i]])))

            weights.append(tf.Variable(
                tf.random_normal([self.hidden_layers[0], self.output_size])))

            return weights

        else:
            #Should not hit this else statement
            return None




if __name__ == '__main__':

    data = [np.array([1,2,3,5]), np.array([4,2,9,2]), np.array([1.2,7.2,3,5])]
    thing = autoencoder(data)

    print thing.encoder_weights
    print thing.decoder_weights
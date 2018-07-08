from image import HyperCube
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class autoencoder():

    def __init__(self, data, learning_rate = 0.01, iterations = 1000, hidden_layers = [10],
        output_size = None, target_outputs = None, batch_size = 32):

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
        self.data_in = tf.placeholder("float", shape=[len(self.data), self.input_size])

        for i, weights in enumerate(self.encoder_weights):
            if i == 0:
                layers.append(tf.nn.sigmoid(tf.add(tf.matmul(self.data_in, weights),
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

    def train_autoencoder(self):
        """
        Use backprop and the training examples in self.data
        to train the network.
        """


        """dataset = self.list_to_dataset()

        for thing in dataset.apply(tf.contrib.data.enumerate_dataset()):
            print thing
            print "-------------------------------------"""

        #x = tf.placeholder(tf.float32, shape=[None, self.input_size])
        y_ = self.decoder

        loss = tf.reduce_mean(tf.pow(self.data_in - y_, 2)) #Use squared error as loss

        input_x = np.stack(self.data)
        input_y = np.stack(self.target_outputs)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        display_step = 1000

        with tf.Session() as sess:

                # Run the initializer
            sess.run(init)
            for i in range(1, self.iterations):

            # Run optimization op (backprop) and cost op (to get loss value)
                _, l = sess.run([optimizer, loss], feed_dict={self.data_in: input_x})
            # Display logs per step
                if i % display_step == 0 or i == 1:
                    print('Step %i: Minibatch Loss: %f' % (i, l))

            #g = sess.run(self.decoder, feed_dict={self.data_in: input_x})

        return l

if __name__ == '__main__':

    data = np.random.randn(100, 50)

    #data = [np.array([2,5.4,17,2.8]), np.array([1.7,9,11,12.2]), np.array([3.1,3,6.8,1]),
    # np.array([1,2.2,3,4])]

    thing = autoencoder(data, hidden_layers = [200], iterations = 10000)

    final_loss = thing.train_autoencoder()

    print "Final loss achieved: " + str(final_loss)



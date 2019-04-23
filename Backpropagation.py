import numpy as np
import time
import matplotlib.pyplot as plt
import pprint

from Data.Data_Parser import binary_to_nparray, label_parser
from Forward import CNN

train_images = binary_to_nparray('C:\\Projects\\CNN\\Data\\train-images.idx3-ubyte', 4)
train_labels = binary_to_nparray('C:\\Projects\\CNN\\Data\\train-labels.idx1-ubyte', 2)

class BackProp(CNN):
    #Create three backpropagation functions to be called upon in varying levels to initialize different weights

    #Get all the arguments passed to it
    def __init__(self, inputs, initialize, batch_size = None, epochs = None):
        super().__init__(inputs, initialize)

    def convolution_backprop(self,output_error, params):
        pass

    def pooling_backprop(self,output_error, params):
        pass

    def fully_connected_backprop(self,output_error, params):
        weights = params['weights']
        biases = params['biases']

        inputs = params['inputs']
        dimensions = params['dimensions']

        der_b2 = []
        der_w2 = []
        for b, w in zip(biases, weights):
            der_b2.append(np.zeros(np.shape(b)))
            der_w2.append(np.zeros(np.shape(w)))

        output = []
        zs = []
        activations = []
        # print(weights)
        for input in inputs:
            holder = input.flatten()
            holder = np.reshape(holder, (np.shape(holder) + (1,)))
            zs_temp = []
            activations_temp = []
            for w, b in zip(weights, biases):
                holder = (np.dot(w, holder))
                zs_temp.append(holder)
                holder = self.sigmoid(holder)
                activations_temp.append(holder)
                # holder += b
            output.append(holder)


        output = np.array(output)

    def std_err_derivative(self, output_activations, y):
        return (output_activations - y)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def der_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def backprop_intializer(self, output, labels):

        self.output_error = self.std_err_derivative(output, labels)

        for layer in self.architecture:
            if layer['operation'] == 'convolution':
                self.convolution_backprop(output_error = self.output_error, params=layer)
            if layer['operation'] == 'pooling':
                self.pooling_backprop(output_error = self.output_error, params=layer)
            if layer['operation'] == 'fully_connected':
                self.fully_connected_backprop(output_error=self.output_error, params=layer)

    def SGD(self):
        pass

    def plotter(self):
        pass


net = BackProp(inputs=train_images[0:2000, : ,:], initialize=True)
output = net.convolution(1, (3,3))
output = net.pooling(3, 'average')
output = net.fully_connected([10, 10])

labels = label_parser(train_labels)
backprop = net.backprop_intializer(output, labels[0:2000, : ,:])
print(backprop)
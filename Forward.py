import numpy as np
import time
import inspect
import matplotlib.pyplot as plt
import pprint

from Data.Data_Parser import binary_to_nparray

t1 = time.clock()

#train_data
train_images = binary_to_nparray('C:\\Projects\\CNN\\Data\\train-images.idx3-ubyte', 4)
train_labels = binary_to_nparray('C:\\Projects\\CNN\\Data\\train-labels.idx1-ubyte', 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class CNN(object):
#To-Do
##Make a Class CNN incorporating these functions
##Parameters can become __init__ and get rid of repetitive code

    def __init__(self, inputs, initialize):

        self.inputs_shape = np.shape(inputs)
        self.inputs = inputs/256
        self.architecture = []

        if initialize == True:
            self.initialize = initialize
        if initialize == False:

            #Write code to read from file
            #Call Functions Here
            pass


    #logic to move filter
    def convolution(self,  stride, filter_shape, inputs = None):

        if inputs == None:
            inputs = self.inputs

        #Currently do not have a padded method available

        self.fil = np.random.ranf((filter_shape))

        output = []
        for idx, input in enumerate(inputs):
            new_input = np.zeros((np.shape(input)))
            input_shape = np.shape(input)

            try:
                len(input_shape) == 2
            except:
                raise ValueError("Input should be a 2 Dimensional Array")

            for r in [i for i in range(0, input_shape[0], stride)]:
                for c in [i for i in range(0, input_shape[1], stride)]:
                    input_space = np.asarray(input[r:r+filter_shape[0], c:c+filter_shape[1]])
                    if np.shape(input_space) == np.shape(self.fil):
                        new_input[r:r+filter_shape[0], c:c+filter_shape[1]] += np.dot(input_space, self.fil)
                    else:
                        #Only look for complete filters, no partials
                        pass

            output.append(new_input)

        output = np.array(output)

        if self.initialize == True:
            convolution_architecture = {}
            convolution_architecture['operation'] = 'convolution'
            convolution_architecture['stride'] = stride
            convolution_architecture['inputs'] = inputs
            convolution_architecture['filter'] = self.fil
            convolution_architecture['output'] = output
            self.architecture.append(convolution_architecture)

        self.inputs = output
        return output


    #pooling logic
    def pooling(self, square_size, type, inputs = None):

        if inputs == None:
            inputs = self.inputs

        bound_size = square_size

        #create a list of multiples of square_size under input size(28)
        #set bounds between these multiples till the maximum which is input size
        ##0-3, 3-6, 6-9, ....
        output = []

        for idx, input in enumerate(inputs):
            input_shape = np.shape(input)[0]
            multiple_list = []
            for i in range(input_shape//bound_size + 1):
                multiple_list.append(i * bound_size)
            del multiple_list[0]
            multiple_list.append(input_shape)

            new_output = np.zeros((len(multiple_list), len(multiple_list)))

            i1 = 0
            # iterations = 0
            for idx, mult in enumerate(multiple_list):
                i2=0
                for idx2, mult2 in enumerate(multiple_list):
                    if type == 'max':
                        new_output[idx, idx2] = np.max(input[i1:mult, i2:mult2].flatten())
                    elif type == 'average':
                        new_output[idx, idx2] = np.sum(input[i1:mult, i2:mult2])/(bound_size**2)
                    else:
                        raise KeyError("The Pooling types accepted are either 'average' or 'max'")
                    i2 = multiple_list[idx2]
                i1 = multiple_list[idx]

            output.append(new_output)

        output = np.array(output)

        if self.initialize == True:
            pooling_architecture = {}
            pooling_architecture['operation'] = 'pooling'
            pooling_architecture['type'] = type
            pooling_architecture['square_size'] = square_size
            pooling_architecture['inputs'] = inputs
            pooling_architecture['output'] = output
            self.architecture.append(pooling_architecture)

        self.inputs = output
        return output


    #initialize network parameters
    def fully_connected(self, dimensions, inputs = None):

        if inputs == None:
            inputs = self.inputs

        weights = []
        biases = []
        # print(biases)
        for a, b in zip(dimensions[1:], dimensions):
            weights.append(np.random.ranf((a, b)))
        for idx, dim in enumerate(dimensions):
            biases.append(np.zeros((dim, 1)))
        # print(biases)


        output = []
        inputs_shape = np.shape(inputs)
        flattened_input = 1
        for dim in inputs_shape[1:]:
            flattened_input *= dim
        shape = (dimensions[0], flattened_input)
        weights.insert(0, np.random.ranf((shape)))

        # print(weights)
        for input in inputs:
            holder = input.flatten()
            holder = np.reshape(holder, (np.shape(holder) + (1, )))
            for w, b in zip(weights, biases):
                holder = (np.dot(w, holder))
                holder = sigmoid(holder)
                #holder += b
            output.append(holder)

        output = np.array(output)

        if self.initialize == True:
            fully_connected_architecture = {}
            fully_connected_architecture['operation'] = 'fully_connected'
            fully_connected_architecture['dimensions'] = dimensions
            fully_connected_architecture['inputs'] = inputs
            fully_connected_architecture['weights'] = weights
            fully_connected_architecture['biases'] = biases
            fully_connected_architecture['output'] = output
            self.architecture.append(fully_connected_architecture)

        self.inputs = output
        return output



net = CNN(inputs=train_images[0:2000,:,:], initialize=True)
output = net.convolution(1, (3,3))
output = net.pooling(3, 'average')
output = net.fully_connected([10, 10])


t2 = time.clock()
t = t2-t1
print(t)
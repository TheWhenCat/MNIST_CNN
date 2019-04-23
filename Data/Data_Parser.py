import numpy as np
import pprint
import struct
import matplotlib.pyplot as plt
import time

import sys
sys.path.append('Data/train-images.idx3-ubyte')

t = time.clock()

def binary_to_nparray(filename, metabytes):
    file = open(filename, 'rb')
    metadata = struct.unpack('>' + 'I' * metabytes,file.read(metabytes*4))
    total_bytes = 1
    shape = ()
    for i in range(len(metadata)):
        if i == 0:
            pass
        else:
            total_bytes *= metadata[i]
            shape += (metadata[i],)
    data = struct.unpack('>' + 'B' * total_bytes, file.read(total_bytes))
    images = np.array(data).reshape(shape)
    return images

def label_parser(labels):
    new_labels = []
    for idx, label in enumerate(labels):
        label_array = np.zeros((10,1))
        label_array[label, 0] = 1
        new_labels.append(label_array)
    new_labels = np.array(new_labels)
    return new_labels


t -= time.clock()
t = -t
print(t)
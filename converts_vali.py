from tokenize import Double
import numpy as np
import pickle as cPickle

path = 'C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/cifar10_data/cifar-10-batches-py/data_batch_1'


fo = open(path, 'rb')
dicts = cPickle.load(fo,encoding='bytes')
fo.close()

data = dicts[b'data']

label = np.array(dicts[b'labels']).reshape(10000,1)

all = np.hstack((data,label))

print(data.shape)
print(label.shape)
print(all.shape)
np.savetxt('C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/cifar10_data/data_batch_1.txt',all,fmt='%0.6f')
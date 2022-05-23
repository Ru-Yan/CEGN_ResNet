from train import Train
import numpy as np
from train import Train
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/验证数据/inner.txt"'''
'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim-all/quiz_batch.txt"'''
ori_path = "C:/Users/lenovo/Desktop/论文模型/cross-resnet-50/sit_data/washed_data/all/test_batch.txt"

dicts = np.loadtxt(ori_path)
data = dicts[:,0:1024]
label = dicts[:,1024:1025]
label = np.reshape(label,(-1))
label = label.astype(int)
data = np.reshape(data,(-1,32,32,1))
print(data.shape)
print(label.shape)
train = Train()
test_image_array = data # Better to be whitened in advance. Shape = [-1, img_height, img_width, img_depth]
predictions = train.test(test_image_array)


batch_size = predictions.shape[0]
in_top1 = predictions - label
num_correct = len(in_top1.nonzero()[0])
print(label.shape)
print((batch_size - num_correct) / float(batch_size))
print(predictions)
print(label)

# predictions is the predicted softmax array.

import tensorflow as tf
import pandas as pd
import tensorflow.compat.v1 as tf_v1
from train import Train
import numpy as np
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

r = np.arange(1,label.shape[0]+1,1)
r = np.reshape(r,(label.shape[0],1))
o = predictions[:,1]
o = np.reshape(o,(label.shape[0],1))
c = label
c = np.reshape(c,(label.shape[0],1))
roc = np.hstack((r,o,c))

roc = pd.DataFrame(roc)

roc.to_csv("roc/test_00.csv", index=False)
# predictions is the predicted softmax array.
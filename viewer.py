import numpy as np  
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import augument
'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/验证数据/ff/a1.txt"'''
'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/biao.txt"'''
'''倾斜正坐 "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/biao_13_35.txt"'''
'''混淆二郎腿"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/lang_15_40.txt"'''
tar_path = "C:/Users/lenovo/Desktop/论文模型/cross-resnet-50/sit_data/washed_data/all/lang.txt"
#tar_path = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/no.txt"
dicts = np.loadtxt(tar_path)
dicts = dicts[:,0:1024]
dicts = dicts.reshape(-1,32,32)
#dicts = np.power(1.05,dicts-57)
dicts1 = augument.fc_aug(dicts)
dicts2 = augument.fc_aug(dicts1)
dicts3 = augument.d_aug(dicts2)


'''
k=10
plt.subplot(1,1,1)
plt.imshow(dicts7[k], cmap=plt.cm.Blues)
plt.show()

'''
k=150
plt.subplot(1,2,1)
plt.imshow(dicts[k], cmap=plt.cm.cool)
plt.subplot(1,2,2)
plt.imshow(dicts3[k], cmap=plt.cm.cool)
plt.show()
'''
plt.imshow(dicts3[k], cmap=plt.cm.Blues)
plt.subplot(2,2,3)
plt.imshow(dicts4[k], cmap=plt.cm.Blues)
plt.subplot(2,2,4)
plt.imshow(dicts7[k], cmap=plt.cm.Blues)
plt.show()
'''
import numpy as np  
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import augument
'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/验证数据/ff/a1.txt"'''
'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/biao.txt"'''
'''倾斜正坐 "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/biao_13_35.txt"'''
'''混淆二郎腿"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/lang_15_40.txt"'''
tar_path = "C:/Users/lenovo/Desktop/论文模型/cross-resnet-50/sit_data/washed_data/all/biao.txt"
#tar_path = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/no.txt"
dicts = np.loadtxt(tar_path)
dicts = dicts[:,0:1024]
dicts = dicts.reshape(-1,32,32)
#dicts = np.power(1.05,dicts-57)
dicts1 = augument.d_aug(dicts)
dicts1 = np.reshape(dicts1,(dicts1.shape[0],-1))
np.savetxt("C:/Users/lenovo/Desktop/论文模型/cross-resnet-50/sit_data/washed_data/all/biao_guass.txt",dicts1,fmt='%.2f',delimiter =' ')
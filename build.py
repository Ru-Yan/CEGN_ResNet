import numpy as np
MAXNUM=10 #设置矩阵元素的最大值
MINNUM=0  #设置矩阵元素的最小值
ROW=1000	#设置矩阵的行数
COL=1025	#设置矩阵的列数
randomMatrix=np.random.randint(MINNUM,MAXNUM,(ROW,COL))
#print(randomMatrix)
np.savetxt(r'C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/cifar10_data/fore_train/test_batch.txt',randomMatrix,fmt="%d", delimiter=' ',footer='By Accelerator')
print(randomMatrix.shape)
import numpy as np

ori = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/验证数据/outer2.txt"
tar = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/验证数据/outer2_quiz.txt"

oo = np.loadtxt(ori)
num_data = oo.shape[0]
order = np.random.permutation(num_data)
oo = oo[order, ...]

print(oo.shape)


np.savetxt(tar,oo,fmt="%d", delimiter=' ')

import numpy as np

ori_1 = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/all/biao.txt"
ori_2 = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/all/lang.txt"
ori_3 = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/all/no.txt"
tar_1 = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/all/data_batch_1.txt"
tar_2 = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/all/data_batch_2.txt"
tar_3 = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/all/data_batch_3.txt"
tar_4 = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/all/data_batch_4.txt"
tar_5 = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/all/data_batch_5.txt"
tar_6 = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/all/test_batch.txt"
tar_7 = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/all/quiz_batch.txt"

ori_1 = np.loadtxt(ori_1)
ori_1 = np.reshape(ori_1,(-1,1025))
ori_1 = ori_1[:,0:1024]
length_1 = ori_1.shape[0]
label_1 = np.full((length_1,1), 0)
ori_1 = np.hstack((ori_1,label_1))
order_1 = np.random.permutation(length_1)
ori_1 = ori_1[order_1, ...]
print(ori_1)
print(ori_1.shape)
ori_1_1 = ori_1[0:1800]
ori_1_2 = ori_1[1800:]


ori_2 = np.loadtxt(ori_2)
ori_2 = np.reshape(ori_2,(-1,1025))
ori_2 = ori_2[:,0:1024]
length_2 = ori_2.shape[0]
label_2 = np.full((length_2,1), 1)
ori_2 = np.hstack((ori_2,label_2))
order_2 = np.random.permutation(length_2)
ori_2 = ori_2[order_2, ...]
print(ori_2)
print(ori_2.shape)
ori_2_1 = ori_2[0:1610]
ori_2_2 = ori_2[1610:]


all = np.vstack((ori_1_1,ori_2_1))
num_data = all.shape[0]
order = np.random.permutation(num_data)
all = all[order, ...]
t_all = np.vstack((ori_1_2,ori_2_2))
num_data = t_all.shape[0]
order = np.random.permutation(num_data)
t_all = t_all[order, ...]
print(all.shape)

b1 = all[0:687]
b2 = all[687:1374]
b3 = all[1374:2061]
b4 = all[2061:2748]
b5 = all[2748:]
t1 = t_all
print(b1.shape)
print(b2.shape)
print(b3.shape)
print(b4.shape)
print(b5.shape)
print(t1.shape)
np.savetxt(tar_1,b1,fmt="%d", delimiter=' ')
np.savetxt(tar_2,b2,fmt="%d", delimiter=' ')
np.savetxt(tar_3,b3,fmt="%d", delimiter=' ')
np.savetxt(tar_4,b4,fmt="%d", delimiter=' ')
np.savetxt(tar_5,b5,fmt="%d", delimiter=' ')
np.savetxt(tar_6,t1,fmt="%d", delimiter=' ')
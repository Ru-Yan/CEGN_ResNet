import numpy as np
import matplotlib.pyplot as plt

ori_path = "C:/Users/lenovo/Desktop/resnet-in-tensorflow-master/sit_data/un_washed_data/无压力/25.txt"
tar_path = "C:/Users/lenovo/Desktop/resnet-in-tensorflow-master/sit_data/washed_data/none/25.txt"

f = open(ori_path,"r")   #设置文件对象

str = f.read()     #将txt文件的所有内容读入到字符串str中
c = str.replace('{','')
c = c.replace('}','')
c = c.replace('"row": 0, "data": ','')
c = c.replace('"row": 1, "data": ','')
c = c.replace('"row": 2, "data": ','')
c = c.replace('"row": 3, "data": ','')
c = c.replace('"row": 4, "data": ','')
c = c.replace('"row": 5, "data": ','')
c = c.replace('"row": 6, "data": ','')
c = c.replace('"row": 7, "data": ','')
c = c.replace('"row": 8, "data": ','')
c = c.replace('"row": 9, "data": ','')
c = c.replace('"row": 10, "data": ','')
c = c.replace('"row": 11, "data": ','')
c = c.replace('"row": 12, "data": ','')
c = c.replace('"row": 13, "data": ','')
c = c.replace('"row": 14, "data": ','')
c = c.replace('"row": 15, "data": ','')
c = c.replace('"row": 16, "data": ','')
c = c.replace('"row": 17, "data": ','')
c = c.replace('"row": 18, "data": ','')
c = c.replace('"row": 19, "data": ','')
c = c.replace('"row": 20, "data": ','')
c = c.replace('"row": 21, "data": ','')
c = c.replace('"row": 22, "data": ','')
c = c.replace('"row": 23, "data": ','')
c = c.replace('"row": 24, "data": ','')
c = c.replace('"row": 25, "data": ','')
c = c.replace('"row": 26, "data": ','')
c = c.replace('"row": 27, "data": ','')
c = c.replace('"row": 28, "data": ','')
c = c.replace('"row": 29, "data": ','')
c = c.replace('"row": 30, "data": ','')
c = c.replace('"row": 31, "data": ','')
c = c.replace(',','')
c = c.replace('[','')
c = c.replace(']','')
c = c.replace('\n',' ')
print(c)
with open(tar_path, "w", encoding='utf-8') as f:
        f.write(c)
f.close()   #将文件关闭
        
dicts = np.loadtxt(tar_path)
dicts = dicts.reshape(-1,32*32)
print(dicts.shape)
length = dicts.shape[0]
data = np.full((length,1), 0)
print(data.shape)
finals = np.hstack((dicts,data))
print(finals.shape)
print(finals)
np.savetxt(tar_path,finals,fmt="%d", delimiter=' ')
plt.imshow(dicts.reshape(-1,32,32)[0], cmap=plt.cm.Blues)
plt.show()
''''''''''''''''''''''''''''''''''''

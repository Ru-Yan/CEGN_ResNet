import matplotlib.pyplot as plt
import numpy as np
import augument

'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/验证数据/ff/a1.txt"'''
'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/biao.txt"'''
'''倾斜正坐 "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/biao_13_35.txt"'''
'''混淆二郎腿"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/lang_15_40.txt"'''
tar_path = "C:/Users/lenovo/Desktop/论文模型/cross-resnet-50/sit_data/washed_data/all/biao.txt"
dicts = np.loadtxt(tar_path)
dicts = dicts[:,0:1024]
dicts = dicts.reshape(-1,32,32)
dicts1 = augument.f_aug(dicts)
dicts2 = augument.f_aug(dicts1)
dicts3 = augument.d_aug(dicts2)

k=567
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
Z = np.array(dicts[k])
x = np.arange(0, len(Z[0]), 1)
y = np.arange(0, len(Z), 1)
X, Y = np.meshgrid(x, y)
#ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
ax = fig.add_subplot(222, projection='3d')
Z = np.array(dicts1[k])
x = np.arange(0, len(Z[0]), 1)
y = np.arange(0, len(Z), 1)
X, Y = np.meshgrid(x, y)
#ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
ax = fig.add_subplot(223, projection='3d')
Z = np.array(dicts2[k])
x = np.arange(0, len(Z[0]), 1)
y = np.arange(0, len(Z), 1)
X, Y = np.meshgrid(x, y)
#ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
ax = fig.add_subplot(224, projection='3d')
Z = np.array(dicts3[k])
x = np.arange(0, len(Z[0]), 1)
y = np.arange(0, len(Z), 1)
X, Y = np.meshgrid(x, y)
#ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()
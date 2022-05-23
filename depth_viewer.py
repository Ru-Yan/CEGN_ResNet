import matplotlib.pyplot as plt
import numpy as np
import augument


def ToGraph(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Z = np.array(data)
    x = np.arange(0,len(Z[0]),1)
    y = np.arange(0,len(Z),1)
    X,Y = np.meshgrid(x,y)
    #ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1,cmap = plt.get_cmap('rainbow'))
    plt.show()

'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/验证数据/ff/a1.txt"'''
'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/biao.txt"'''
'''倾斜正坐 "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/biao_13_35.txt"'''
'''混淆二郎腿"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim/lang_15_40.txt"'''
tar_path = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/washed_data/2/二郎腿-1-trim.txt"
k = 1
dicts = np.loadtxt(tar_path)
dicts = dicts[:,0:1024]
dicts = dicts.reshape(-1,32,32)
#dicts = np.power(1.05,dicts-57)
dicts1 = augument.fc_aug(dicts)
dicts2 = augument.fc_aug(dicts1)
dicts3 = augument.d_aug(dicts2)
ToGraph(dicts[k])
ToGraph(dicts3[k])
import spidev
import time
import numpy as np
from test_matrix import test_matrix
import augument
import matplotlib.pyplot as plt

#初始化设备
decay = 0.15
spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 7629
time.sleep(decay)#等待 stm32 0.3s(300ms)

#line_id 为当前行号
line_id = 0
#is_busy为状态标识，0=错误，1=空闲，2=正在采集，3=完成采集
is_busy = 0
#is_collect为是否完成采集
is_collect = 0
#sit_matrix为保存的坐姿数据
sit_matrix = np.zeros((32,32))
#single_data为一次采集过程中接受的一帧数据信息
single_data = np.zeros((1,50))
#is_success为是否接收成功
is_success = 0


print("Wait For Answer")
spi.writebytes([0xff])
time.sleep(decay)#等待 stm32 0.3s(300ms)
is_busy = spi.readbytes(1)
print(is_busy)
spi.writebytes([0xfe])
time.sleep(decay)#等待 stm32 0.3s(300ms)
is_collect = spi.readbytes(1)
print(is_collect)
    
for i in range(32):    
    single_data = spi.readbytes(50)
    print(single_data)
    sit_matrix[i] = single_data[4:36]
    spi.writebytes([0xfc])
    time.sleep(decay)#等待 stm32 0.3s(300ms)
    is_success = spi.readbytes(1)
    print(is_success)

sit_matrix = sit_matrix.reshape((1,32,32,1))
print(sit_matrix)
print(test_matrix(sit_matrix))

sit_matrix = sit_matrix.reshape(32,32)
sit_matrix = augument.aug(sit_matrix)
plt.imshow(sit_matrix, cmap=plt.cm.Blues)
plt.show()

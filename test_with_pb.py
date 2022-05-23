import tensorflow as tf
import os
import tensorflow.compat.v1 as tf_v1
from tensorflow.python.platform import gfile
FLAGS = tf_v1.app.flags.FLAGS
import cv2
import numpy as np
import augument
import time

'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/验证数据/inner.txt"'''
'''"C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim-all/quiz_batch.txt"'''
ori_path = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/sit_data/trim-all/data_batch_1.txt"
model_path = "logs_test_110/model.ckpt-2000.tflite"
dicts = np.loadtxt(ori_path)
data = dicts[:,0:1024]
label = dicts[:,1024:1025]
label = np.reshape(label,(-1))
label = label.astype(int)
data = np.reshape(data,(-1,32,32,1))
data = data.astype(np.float32)
data = augument.aug(data)
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
print(str(input_details))
output_details = interpreter.get_output_details()
print(str(output_details))

#with tf.Session( ) as sess:
if 1:
    model_interpreter_time = 0
    start_time = time.time()
    # 填装数据
    model_interpreter_start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], data)
        
    # 注意注意，我要调用模型了
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.argmax(predictions, axis=1)
    model_interpreter_time += time.time() - model_interpreter_start_time
        
    batch_size = predictions.shape[0]
    in_top1 = predictions - label
    num_correct = len(in_top1.nonzero()[0])
    print(label.shape)
    print((batch_size - num_correct) / float(batch_size))
    print(predictions)
    print(label) 
used_time = time.time() - start_time
print('used_time:{}'.format(used_time))
print('model_interpreter_time:{}'.format(model_interpreter_time))

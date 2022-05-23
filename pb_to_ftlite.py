import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
#把pb文件路径改成自己的pb文件路径即可
path = "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/logs_test_110/model.ckpt-2000.pb"
 
#如果是不知道自己的模型的输入输出节点，建议用tensorboard做可视化查看计算图，计算图里有输入输出的节点名称
inputs = ["Placeholder"]
outputs = ["fc/add"]
#转换pb模型到tflite模型
converter = tf_v1.lite.TFLiteConverter.from_frozen_graph(path, inputs, outputs,{"Placeholder" : [1, 32, 32, 1]})
converter.post_training_quantize = True
tflite_model = converter.convert()
#yolov3-tiny_160000.tflite这里改成自己想要保存tflite模型的地址即可
open("model.ckpt-2000.tflite", "wb").write(tflite_model)
 
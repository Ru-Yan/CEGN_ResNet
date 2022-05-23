from train import Train
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import time

def test_matrix(matrix):
    start_time = time.time()
    data = matrix
    print(data.shape)
    train = Train()
    test_image_array = data # Better to be whitened in advance. Shape = [-1, img_height, img_width, img_depth]
    predictions = train.test(test_image_array)
    batch_size = predictions.shape[0]
    used_time = time.time() - start_time
    print('used_time:{}'.format(used_time))
    print(predictions)
    return (predictions)
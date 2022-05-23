import tarfile
from six.moves import urllib
import sys
import numpy as np
import pickle as cPickle
import os
import cv2

data_dir = 'sit_data/washed_data/all'
full_data_dir = 'sit_data/washed_data/all/data_batch_'
vali_dir = 'sit_data/washed_data/all/test_batch'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_CLASS = 3

TRAIN_RANDOM_LABEL = False # Want to use random label for train data?
VALI_RANDOM_LABEL = False # Want to use random label for validation?

NUM_TRAIN_BATCH = 5 # How many batches of files you want to read in, from 0 to 5)
EPOCH_SIZE = 200 * NUM_TRAIN_BATCH
TEST_SIZE = 250


def maybe_download_and_extract():
    '''
    用cifa10进行测试训练，下载cifar10
    '''
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size)
                                                             / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _read_one_batch(path, is_random_label):
    '''
    测试训练数据一共有五个batch。验证集有一个batch。此函数将调用一组batch，然后将图像和对应标签转成numpy格式

    param path: 一组batch的地址
    param is_random_label: 是否随机化标签
    return: 数据和标签的numpy数组
    '''
    dicts = np.loadtxt(path+'.txt')
    dicts.reshape(-1,1025)
    data = dicts[...,0:1024]
    if is_random_label is False:
        label = np.array(dicts[...,-1])
    else:
        labels = np.random.randint(low=0, high=255, size=1000)
        label = np.array(labels)
    return data, label


def read_in_all_images(address_list, shuffle=True, is_random_label = False):
    """
    读取所有训练集或验证集，并可以随机化，然后返回数据和标签的numpy数组

    param address_list: cPickle文件的list
    return:数据和标签的四维numpy数组: [num_images,
    image_height, image_width, image_depth] 和 [num_images]
    """
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT])
    label = np.array([])

    for address in address_list:
        print ('Reading images from ' + address)
        batch_data, batch_label = _read_one_batch(address, is_random_label)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH,1))


    if shuffle is True:
        print ('Shuffling')
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label


def horizontal_flip(image, axis):
    '''
    数据增强--翻转图像
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image


def whitening_image(image_np):
    '''
    数据增强--白化
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np


def random_crop_and_flip(batch_data, padding_size):
    '''
    数据增强--随机裁剪和翻转
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH,1)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1).reshape(-1,IMG_HEIGHT,IMG_WIDTH,1)

    return cropped_batch


def prepare_train_data(padding_size):
    '''
    将读取到的数据转成numpy格式然后padding
    param padding_size: padding层数
    return: 所有数据以及相对应的标签
    '''
    path_list = []
    for i in range(1, NUM_TRAIN_BATCH+1):
        path_list.append(full_data_dir + str(i))
    data, label = read_in_all_images(path_list, is_random_label=TRAIN_RANDOM_LABEL)
    
    return data, label


def read_validation_data():
    '''
    读取验证集，同时白化
    '''
    validation_array, validation_labels = read_in_all_images([vali_dir],
                                                       is_random_label=VALI_RANDOM_LABEL)
  

    return validation_array, validation_labels



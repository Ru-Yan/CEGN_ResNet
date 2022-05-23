import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageOps
import math
import pylab
import matplotlib.pyplot as plt


def NormalizeImage(img, mean=[0.456],
                   std=[1]):
    pic = img
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]
    pic = pic / 255.0
    pic -= mean
    pic /= std
    return pic

def RandomFlip(im,prob = 0.5):
    probs = np.random.uniform(0, 1)
    if probs >= prob:
        return im
    num = np.size(im, 0)
    resaot = im
    for i in range(num):
        resaot[i] = np.fliplr(im[i])
    return resaot

def RandomDistort(im, brightness_lower=0.5,
                  brightness_upper=1.5,
                  contrast_lower=0.5,
                  contrast_upper=1.5,
                  saturation_lower=0.5,
                  saturation_upper=1.5,
                  brightness_prob=1,
                  contrast_prob=1,
                  saturation_prob=1,
                  ):
    images = np.concatenate((im, im, im), axis=3)
    num = np.size(im, 0)
    resaot = images
    for i in range(num):
        img = Image.fromarray(np.uint8(images[i]))
        brightness_delta = np.random.uniform(brightness_lower,
                                             brightness_upper)
        prob = np.random.uniform(0, 1)
        if prob < brightness_prob:
            img = ImageEnhance.Brightness(img).enhance(brightness_delta)

        contrast_delta = np.random.uniform(contrast_lower,
                                           contrast_upper)
        prob = np.random.uniform(0, 1)
        if prob < contrast_prob:
            img = ImageEnhance.Contrast(img).enhance(contrast_delta)

        saturation_delta = np.random.uniform(saturation_lower,
                                             saturation_upper)
        prob = np.random.uniform(0, 1)
        if prob < saturation_prob:
            img = ImageEnhance.Color(img).enhance(saturation_delta)

        resaot[i] = np.array(img)

    return resaot[:, :, :, 0]


def imrotate(image, rotate_prob, rotate_left=-20, rotate_right=20):
    prob = np.random.uniform(0, 1)
    if prob >= rotate_prob:
        return image
    b = np.random.uniform(rotate_left, rotate_right)
    b = -math.radians(b % 360)  # 将角度化为弧度
    n = np.size(image, 1)
    m = np.size(image, 2)
    center = (n/2.0, m/2.0)
    img = image
    num = np.size(image, 0)
    nn = n
    nm = m

    def inmap(x, y):
        return True if x >= 0 and x < n and y >= 0 and y < m else False

    for id in range(num):
        # 反向推
        for x in range(nn):
            for y in range(nm):
                x0 = (x-center[0])*math.cos(-b) + \
                    (y-center[1])*math.sin(-b)+center[0]
                y0 = -1*(x-center[0])*math.sin(-b) + \
                    (y-center[1])*math.cos(-b)+center[1]
                # 将坐标对齐
                x0 = x0-(nn-n)/2
                y0 = y0-(nm-m)/2
                # 双线性内插值
                i = int(x0)
                j = int(y0)
                u = x0 - i
                v = y0 - j
                img[id][x][y] = 0
                if inmap(i, j):
                    f1 = (1-u)*(1-v)*image[id][i][j]
                    img[id][x][y] += f1
                    if inmap(i, j+1):
                        f2 = (1-u)*v*image[id][i][j+1]
                        img[id][x][y] += f2
                    if inmap(i+1, j):
                        f3 = u*(1-v)*image[id][i+1][j]
                        img[id][x][y] += f3
                    if inmap(i+1, j+1):
                        f4 = u*v*image[id][i+1][j+1]
                        img[id][x][y] += f4
    return img

def aug(image):
    image = np.where(image>59, np.log(image-58)/np.log(5), 0)
    return image

def aug_constr(image):
    image = np.where(image>60, np.log(np.power(1.25,image-60))/np.log(1.3), 0)
    return image

def t_f_aug(image):
    p_image = np.pad(image,((0,0),(2,2),(2,2),(0,0)),constant_values=(58,58))
    #avg_bool = np.ones(image.shape, dtype=bool)
    avg_double = np.ones(image.shape)
    for i in range(32):
        for j in range(32):
            avg=p_image[...,i+2,j+2,0]+p_image[...,i+2,j+1,0]+p_image[...,i+2,j,0]+p_image[...,i+2,j+3,0]+p_image[...,i+2,j+4,0]+p_image[...,i+1,j+2,0]+p_image[...,i,j+2,0]+p_image[...,i+3,j+2,0]+p_image[...,i+4,j+2,0]
            avg /= 9.0
            #avg1 = avg>59.8
            #avg_bool[...,i,j] = avg1
            avg_double[...,i,j,0] = avg
    #image = np.where(avg_bool,image,0)
    return avg_double

def f_aug(image):
    p_image = np.pad(image,((0,0),(2,2),(2,2)),constant_values=(58,58))
    #avg_bool = np.ones(image.shape, dtype=bool)
    avg_double = np.ones(image.shape)
    for i in range(32):
        for j in range(32):
            avg=p_image[...,i+2,j+2]+p_image[...,i+2,j+1]+p_image[...,i+2,j]+p_image[...,i+2,j+3]+p_image[...,i+2,j+4]+p_image[...,i+1,j+2]+p_image[...,i,j+2]+p_image[...,i+3,j+2]+p_image[...,i+4,j+2]
            avg /= 9.0
            #avg1 = avg>59.8
            #avg_bool[...,i,j] = avg1
            avg_double[...,i,j] = avg
    #image = np.where(avg_bool,image,0)
    return avg_double

'''def f_aug(image):
    p_image = np.pad(image,((0,0),(2,2),(2,2)),constant_values=(58,58))
    #avg_bool = np.ones(image.shape, dtype=bool)
    avg_double = np.ones(image.shape)
    for i in range(32):
        for j in range(32):
            avg=p_image[...,i+2,j+2]+p_image[...,i+2,j+1]+p_image[...,i+2,j]+p_image[...,i+2,j+3]+p_image[...,i+2,j+4]+p_image[...,i+1,j+2]+p_image[...,i,j+2]+p_image[...,i+3,j+2]+p_image[...,i+4,j+2]
            avg /= 9.0
            #avg1 = avg>59.8
            #avg_bool[...,i,j] = avg1
            avg_double[...,i,j] = avg
    #image = np.where(avg_bool,image,0)
    return avg_double'''

def fc_aug(image):
    p_image = np.pad(image,((0,0),(2,2),(2,2)),constant_values=(58,58))
    #avg_bool = np.ones(image.shape, dtype=bool)
    avg_double = np.ones(image.shape)
    for i in range(32):
        for j in range(32):
            avg=p_image[...,i,j]+p_image[...,i+1,j+1]+p_image[...,i+2,j+2]+p_image[...,i+3,j+3]+p_image[...,i+4,j+4]+p_image[...,i,j+4]+p_image[...,i+1,j+3]+p_image[...,i+3,j+1]+p_image[...,i+4,j]
            avg /= 9.0
            #avg1 = avg>59.8
            #avg_bool[...,i,j] = avg1
            avg_double[...,i,j] = avg
    #image = np.where(avg_bool,image,0)
    return avg_double

def ff_aug(image):
    p_image = np.pad(image,((0,0),(2,2),(2,2)),constant_values=(58,58))
    #avg_bool = np.ones(image.shape, dtype=bool)
    avg_double = np.ones(image.shape)
    for i in range(32):
        for j in range(32):
            avg=p_image[...,i,j]+4*p_image[...,i,j+1]+7*p_image[...,i,j+2]+4*p_image[...,i,j+3]+p_image[...,i,j+4]+4*p_image[...,i+1,j]+16*p_image[...,i+1,j+1]+26*p_image[...,i+1,j+2]+16*p_image[...,i+1,j+3]+4*p_image[...,i+1,j+4]+7*p_image[...,i+2,j]+26*p_image[...,i+2,j+1]+41*p_image[...,i+2,j+2]+26*p_image[...,i+2,j+3]+7*p_image[...,i+2,j+4]+4*p_image[...,i+3,j]+16*p_image[...,i+3,j+1]+26*p_image[...,i+3,j+2]+16*p_image[...,i+3,j+3]+4*p_image[...,i+3,j+4]+p_image[...,i+4,j]+4*p_image[...,i+4,j+1]+7*p_image[...,i+4,j+2]+4*p_image[...,i+4,j+3]+p_image[...,i+4,j+4]
            avg /= 273.0
            #avg1 = avg>59.8
            #avg_bool[...,i,j] = avg1
            avg_double[...,i,j] = avg
    #image = np.where(avg_bool,image,0)
    return avg_double

def d_aug(image):
    image = np.where(image>67, np.log(image-53)/np.log(1.2)+52.53, image)
    return image
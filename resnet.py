'''
残差网络结构
'''
from input import NUM_CLASS
import numpy as np
from hyper_parameters import *

tf_v1.disable_eager_execution()

BN_EPSILON = 0.001

def activation_summary(x):
    '''
    x: 张量
    return: 添加 histogram summary和scalar summary两种统计信息
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.keras.initializers.RandomNormal, is_fc_layer=False):
    '''
    name: string. 变量名
    shape: 维度
    initializer:初始化
    is_fc_layer: 是否是全连接层
    return: 创建的变量
    '''
    
    regularizer = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
    
    new_variables = tf_v1.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    input_layer: 二维张量
    num_labels: 标签数目
    return: Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf_v1.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h

def data_attention(input_layer):
    N,H,W,C=input_layer.shape
    Avgpool = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 1, 1, 1], padding='SAME')
    Maxpool = tf.nn.max_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 1, 1, 1], padding='SAME')
    avg_conv = tf.nn.conv2d(Avgpool, filters=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    max_conv = tf.nn.conv2d(Maxpool, filters=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    add = avg_conv + max_conv
    acti = tf.nn.sigmoid(add)
    return acti*input_layer

def GroupNorm(x,G=16,eps=1e-5):
    N,H,W,C=x.shape    	
    x=tf.reshape(x,[tf.cast(N,tf.int32),tf.cast(H,tf.int32),tf.cast(W,tf.int32),tf.cast(G,tf.int32),tf.cast(C//G,tf.int32)])
    mean,var=tf_v1.nn.moments(x,[1,2,4],keep_dims=True)
    x=(x-mean)/tf.sqrt(var+eps)
    x=tf.reshape(x,[tf.cast(N,tf.int32),tf.cast(H,tf.int32),tf.cast(W,tf.int32),tf.cast(C,tf.int32)])
    gamma = tf.Variable(tf.ones(shape=[1,1,1,tf.cast(C,tf.int32)]), name="gamma")
    beta = tf.Variable(tf.zeros(shape=[1,1,1,tf.cast(C,tf.int32)]), name="beta")
    return x*gamma+beta


def batch_normalization_layer(input_layer, dimension):
    '''
    批标准化
    input_layer: 四维张量
    param dimension: input_layer.get_shape().as_list()[-1]
    '''
    bn_layer = GroupNorm(input_layer)
    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    卷积+批标准化+relu
    input_layer: 四维张量
    filter_shape: [filter_height, filter_width, filter_depth, filter_number]
    stride: 卷积步长
    :return: 四维张量 Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    批标准化+relu+卷积
    input_layer: 四维张量
    filter_shape: [filter_height, filter_width, filter_depth, filter_number]
    stride: 卷积步长
    return: 四维张量 Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block(input_layer, output_channel, first_block=False):
    '''
    残差块
    input_layer: 四维张量
    output_channel: return_tensor.get_shape().as_list()[-1]
    first_block: 这个残差块是否是网络第一个残差块
    return: 四维张量
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf_v1.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filters=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf_v1.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def inference(input_tensor_batch, n, reuse):
    '''
    轻量型残差网络定义. 总层数 = 1 + 2n + 2n + 2n +1 = 6n + 2
    input_tensor_batch: 四维的张量
    n: 残差块数量
    reuse: 如果要创建训练集, reuse=False. 如果要创建验证集且使用训练集权重, resue=True
    return: 网络的最后一层，不是softmax后的结果
    '''

    layers = []
    layers.append(data_attention)
    with tf_v1.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 1, 16], 1)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf_v1.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 16)
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf_v1.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32)
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf_v1.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf_v1.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, NUM_CLASS)
        layers.append(output)

    return layers[-1]


def test_graph(train_dir='logs'):
    '''
    运行这个来在tensorboard上看到网络架构
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 1]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

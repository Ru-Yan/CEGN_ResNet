from resnet import *
from datetime import datetime
import time
from input import *
import pandas as pd
import augument

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存


tf_v1.disable_eager_execution()
class Train(object):
    '''
    This Object is responsible for all the training and validation process
    '''
    def __init__(self):
        # Set up all the placeholders
        self.placeholders()

    #构建占位符
    def placeholders(self):
        #图片占位符
        self.image_placeholder = tf_v1.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH,1])
        #标签占位符
        self.label_placeholder = tf_v1.placeholder(dtype=tf.int32, shape=[FLAGS.train_batch_size])
        #验证集图片占位符
        self.vali_image_placeholder = tf_v1.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                IMG_HEIGHT, IMG_WIDTH,1])
        #验证集标签占位符
        self.vali_label_placeholder = tf_v1.placeholder(dtype=tf.int32, shape=[FLAGS.validation_batch_size])
        #学习率占位符
        self.lr_placeholder = tf_v1.placeholder(dtype=tf.float32, shape=[])

        return (self.image_placeholder,self.label_placeholder,self.vali_image_placeholder,self.vali_label_placeholder,self.lr_placeholder)



    #构建图
    def build_train_validation_graph(self):
        #全局步数
        global_step = tf.Variable(0, trainable=False)
        #验证集步数
        validation_step = tf.Variable(0, trainable=False)

        #训练集
        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        #验证集
        vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True)

        #正则化
        regu_losses = tf_v1.get_collection(tf_v1.GraphKeys.REGULARIZATION_LOSSES)
        #交叉熵
        loss = self.loss(logits, self.label_placeholder)
        #总和
        self.full_loss = tf.add_n([loss] + regu_losses)
        #预测
        predictions = tf.nn.softmax(logits)
        #取得结果
        self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)

        #旋转左右区间（角度）
        rotate_left=-20
        rotate_right=20
        # 验证集损失
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        #验证集预测
        vali_predictions = tf.nn.softmax(vali_logits)
        #取得结果
        self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)
        #定义训练集操作，ema操作，验证集操作
        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error)
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)



    def train(self):
        '''
        训练主函数
        '''

        # 加载训练数据
        all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
        vali_data, vali_labels = read_validation_data()

        # 构建训练集的图和验证集的图
        self.build_train_validation_graph()

        #保存训练节点
        saver = tf_v1.train.Saver(tf_v1.global_variables(),max_to_keep=200)
        summary_op = tf_v1.summary.merge_all()
        init = tf_v1.initialize_all_variables()
        sess = tf_v1.Session(config=config) 

        # 恢复训练节点
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print('正在恢复训练节点')
        else:
            sess.run(init)

        # 总结写到tensorboard
        summary_writer = tf_v1.summary.FileWriter(train_dir, sess.graph)


        #保存csv文件需要的参数
        step_list = []
        train_error_list = []
        val_error_list = []
        loss_list = []

        print('开始训练坐姿识别网络...')
        print('----------------------------')

        for step in range(FLAGS.train_steps):

            train_batch_data, train_batch_labels = self.generate_augment_train_batch(all_data, all_labels,
                                                                        FLAGS.train_batch_size)
            train_batch_data = np.reshape(train_batch_data,(-1,32,32,1))

            validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data,
                                                           vali_labels, FLAGS.validation_batch_size)

            if step % FLAGS.report_freq == 0:

                if FLAGS.is_full_validation is True:
                    validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                            top1_error=self.vali_top1_error, vali_data=vali_data,
                                            vali_labels=vali_labels, session=sess)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                        simple_value=validation_error_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                else:
                    _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                     self.vali_top1_error,
                                                                 self.vali_loss],
                                                {self.vali_image_placeholder: validation_batch_data,
                                                 self.vali_label_placeholder: validation_batch_labels,
                                                 self.lr_placeholder: FLAGS.init_lr})

                val_error_list.append(validation_error_value)


            start_time = time.time()

            _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                           self.full_loss, self.train_top1_error],
                                {self.image_placeholder: train_batch_data,
                                  self.label_placeholder: train_batch_labels,
                                  self.vali_image_placeholder: validation_batch_data,
                                  self.vali_label_placeholder: validation_batch_labels,
                                  self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time


            if step % FLAGS.report_freq == 0:
                

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                                    sec_per_batch))
                print('Train top1 error = ', train_error_value)
                print('Validation top1 error = %.4f' % validation_error_value)
                print('验证集loss = ', validation_loss_value)
                print('----------------------------')

                step_list.append(step)
                train_error_list.append(train_error_value)
                loss_list.append(train_loss_value)



            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print('学习率衰减到 ',FLAGS.init_lr)

            # Save checkpoints every 10000 steps
            if step % FLAGS.report_freq == 0 or (step + 1) == FLAGS.train_steps:
                print('save model')
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                'validation_error': val_error_list,'loss':loss_list})
                df.to_csv(train_dir + FLAGS.version + '_error.csv')


    def test(self, test_image_array):
        '''
        测试集
        param test_image_array:四维numpy数组 [num_test_images, img_height, img_width,
        img_depth]
        return: softmax后的测试结果： [num_test_images, num_labels]
        '''
        num_test_images = test_image_array.shape[0]
        num_batches = num_test_images // FLAGS.validation_batch_size
        remain_images = num_test_images % FLAGS.validation_batch_size
        test_image_array = augument.t_f_aug(test_image_array)
        test_image_array = augument.t_f_aug(test_image_array)
        test_image_array = augument.d_aug(test_image_array)
        print('总共有%i个test batch...' %num_batches)

        # 创建占位符
        self.test_image_placeholder = tf_v1.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.validation_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH,1])
                                    

        # 测试集网络
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        predictions = tf.nn.softmax(logits)

        # 初始换新session和checkpoint
        saver = tf_v1.train.Saver(tf_v1.all_variables())
        sess = tf_v1.Session(config=config)

        saver.restore(sess, FLAGS.test_ckpt_path)
        print('模型调用地址 ', FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # 根据batch进行加载
        for step in range(num_batches):
            if step % 10 == 0:
                print('%i组batch加载完成!' %step)
            offset = step * FLAGS.validation_batch_size
            test_image_batch = test_image_array[offset:offset+FLAGS.validation_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: test_image_batch})
            

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # 取余数后剩下的数据的加载
        if remain_images != 0:
            self.test_image_placeholder = tf_v1.placeholder(dtype=tf.float32, shape=[remain_images,
                                                        IMG_HEIGHT, IMG_WIDTH,1])
            # 建立测试集图
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))
        print(prediction_array)
        #prediction_array = np.argmax(prediction_array, axis=1)
        return prediction_array



    ## 要用到的函数
    def loss(self, logits, labels):
        '''
        计算loss
        '''
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean


    def top_k_error(self, predictions, labels, k):
        '''
        计算前k个错误
        predictions: 二维的张量[batch_size, num_labels]
        labels:一维的张量[batch_size, 1]
        k: int
        return:0维张量[1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf_v1.to_float(tf_v1.math.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)


    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        数据增强：batch
        '''
        offset = np.random.choice(TEST_SIZE - vali_batch_size, 1)[0]
        vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
        #vali_data_batch = augument.t_f_aug(vali_data_batch)
        #vali_data_batch = augument.t_f_aug(vali_data_batch)
        #vali_data_batch = augument.d_aug(vali_data_batch)
        vali_label_batch = vali_label[offset:offset+vali_batch_size]
        return vali_data_batch, vali_label_batch


    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        数据增强：generate a batch of train data, and random crop, horizontally flip，whitening_image
        '''
        global mat_f1,mat_f2,mat_f3,mat_f4,mat_i,mat_j
        offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset+train_batch_size, ...]
        batch_data = augument.RandomFlip(batch_data)
        #batch_data = augument.t_f_aug(batch_data)
        #batch_data = augument.t_f_aug(batch_data)
        #batch_data = augument.d_aug(batch_data)
        batch_label = train_labels[offset:offset+FLAGS.train_batch_size]

        return batch_data, batch_label


    def train_operation(self, global_step, total_loss, top1_error):
        '''
        定义训练操作
        param global_step:0维张量
        param total_loss:0维张量
        param top1_error:0维张量
        return: 两个操作. train_op，train_ema_op
        '''
        #添加训练集参数到tensorboard
        #tf.summary.scalar('learning_rate', self.lr_placeholder)
        #tf.summary.scalar('train_loss', total_loss)
        #tf.summary.scalar('train_top1_error', top1_error)

        #添加ema参数到tensorboard
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        #tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        #tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        opt = tf_v1.train.AdamOptimizer(learning_rate=self.lr_placeholder)
        train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        #top1_error_val = ema.average(top1_error)
        #top1_error_avg = ema2.average(top1_error)
        #loss_val = ema.average(loss)
        #loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        #tf.summary.scalar('val_top1_error', top1_error_val)
        #tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        #tf.summary.scalar('val_loss', loss_val)
        #tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        进行10000张数据验证
        Runs validation on all the 10000 valdiation images
        param loss: 0维张量
        :param top1_error: 0维张量
        :param session: 当前session
        :param vali_data: 四维numpy数组
        :param vali_labels: 一维numpy数组
        :param batch_data: 四维batch化numpy数组.
        :param batch_label: 一维batch化numpy数组.
        :return: loss的list，错误的list
        '''
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)




if __name__ == '__main__':
    # 初始化
    train = Train()
    # 开始训练
    train.train()


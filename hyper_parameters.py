import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

FLAGS = tf_v1.app.flags.FLAGS


##保存路径，tensorboard输出，终端输出

tf_v1.app.flags.DEFINE_string('version', 'test_110', '''A version number defining the directory to save
logs and checkpoints''')
tf_v1.app.flags.DEFINE_integer('report_freq', 20, '''Steps takes to output errors on the screen
and write summaries''')
tf_v1.app.flags.DEFINE_float('train_ema_decay', 0.95, '''The decay factor of the train error's
moving average shown on tensorboard''')


##超参数

tf_v1.app.flags.DEFINE_integer('train_steps', 20000, '''Total steps that you want to train''')
tf_v1.app.flags.DEFINE_boolean('is_full_validation', False, '''Validation w/ full validation set or
a random batch''')
tf_v1.app.flags.DEFINE_integer('train_batch_size', 256, '''Train batch size''')
tf_v1.app.flags.DEFINE_integer('validation_batch_size',1, '''Validation batch size, better to be
a divisor of 10000 for this task''')
tf_v1.app.flags.DEFINE_integer('test_batch_size', 32, '''Test batch size''')

tf_v1.app.flags.DEFINE_float('init_lr', 0.0001, '''Initial learning rate''')
tf_v1.app.flags.DEFINE_float('lr_decay_factor', 0.1, '''How much to decay the learning rate each
time''')
tf_v1.app.flags.DEFINE_integer('decay_step0', 10000, '''At which step to decay the learning rate''')
tf_v1.app.flags.DEFINE_integer('decay_step1', 50000, '''At which step to decay the learning rate''')

tf_v1.app.flags.DEFINE_integer('num_residual_blocks', 8, '''How many residual blocks do you want''')
tf_v1.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')

tf_v1.app.flags.DEFINE_integer('padding_size', 2, '''In data augmentation, layers of zero padding on
each side of the image''')


##断点训练

tf_v1.app.flags.DEFINE_string('ckpt_path', 'C:/Users/lenovo/Desktop/论文模型/cross-resnet-50/logs_test_110/model.ckpt-160', '''Checkpoint
directory to restore''')
tf_v1.app.flags.DEFINE_boolean('is_use_ckpt',False, '''Whether to load a checkpoint and continue
training''')

tf_v1.app.flags.DEFINE_string('test_ckpt_path', 'C:/Users/lenovo/Desktop/论文模型/cross-resnet-50/logs_test_110/model.ckpt-140', '''Checkpoint
directory to restore''')


train_dir = 'logs_' + FLAGS.version + '/'

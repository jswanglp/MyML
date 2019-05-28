# 该程序实现 TensorFlow 下的三层感知机对 mnist 的分类
# 定义了可以在 cmd 下输出红色字体的存储记录，定义了实时进度条显示完成度的 print_progress 函数
# 采取初始 learning rate 每 16 步衰减为 0.95 倍的策略
# 采用 mini-batch SGD 的梯度训练方法
# conding:utf-8
import numpy as np
import os, sys, ctypes
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from progress_bar import print_progress

# -----------------------定义能在 cmd 命令下输出红色字体的函数 printRed()---------------------------
STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12

FOREGROUND_RED = 0x0c # red.
FOREGROUND_BLUE = 0x09 # blue.
FOREGROUND_GREEN = 0x0a # green.

std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

def set_cmd_text_color(color, handle=std_out_handle):
    Bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
    return Bool

def resetColor():
    set_cmd_text_color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)

def printRed(mess):
    set_cmd_text_color(FOREGROUND_RED)
    sys.stdout.write(mess)
    resetColor()

# printRed("Model restored ... ...\n") # 测试
# print("Model restored ... ...\n")
#--------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # 定义一些必要的 Flags 
    tf.app.flags.DEFINE_string('events_path', os.path.dirname(os.path.abspath(__file__)) + '/Tensorboard', 'Directory where event logs are written to.')
    tf.app.flags.DEFINE_string('checkpoints_path', os.path.dirname(os.path.abspath(__file__)) + '/Checkpoints', 'Directory where checkpoints are written to.')
    
    tf.app.flags.DEFINE_integer('max_num_checkpoints', 3, 'Maximum number of checkpoints that TensorFlow will keep.')
    tf.app.flags.DEFINE_integer('num_class', 10, 'Number of classes.')
    tf.app.flags.DEFINE_integer('batch_size', 256, 'Number of model batchsize.')
    tf.app.flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs for training.')
    tf.app.flags.DEFINE_integer('num_neurons', 250, 'Number of neurons in hidden layers.')
    tf.app.flags.DEFINE_integer('num_batch', 100, 'Number of batchs in one epoch.') # 每个 epoch 中包含的 mini-batch 训练的次数，应能被 num_epochs整除

    tf.app.flags.DEFINE_float('learning_rate', 5e-3, 'Initial learning rate.')
    tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')
    tf.app.flags.DEFINE_float('decay_steps', 16., 'Decay steps')
    tf.app.flags.DEFINE_float('keep_prob', 0.5, 'The probability that each element is kept.')
    
    tf.app.flags.DEFINE_boolean('online_test', True, 'Online test or not ?')

    FLAGS = tf.app.flags.FLAGS

    # events 和 checkpoints 的存储位置的确认
    if not os.path.isabs(FLAGS.events_path):
        raise ValueError('You must assign absolute path for --events_path')

    if not os.path.isabs(FLAGS.checkpoints_path):
        raise ValueError('You must assign absolute path for --checkpoints_path')

    # 训练、测试用的图像与标签的提取
    mnist = input_data.read_data_sets("sample_data/MNIST_data", reshape=True, one_hot=True)
    train_image = mnist.train.images
    train_label = mnist.train.labels
    test_image = mnist.test.images
    test_label = mnist.test.labels

    num_samples, num_features = train_image.shape

    # graph 的建立
    graph = tf.Graph()
    with graph.as_default():

        # learning rate 的选取策略：        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        decay_steps = FLAGS.decay_steps
        learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=decay_steps,
                                                    decay_rate=FLAGS.learning_rate_decay_factor,
                                                    staircase=True,
                                                    name='exponential_decay')
        
        # placeholder 的定义
        image_place = tf.placeholder(tf.float32, shape=[None, num_features], name='image')
        label_place = tf.placeholder(tf.float32, shape=[None, FLAGS.num_class], name='label')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 三层感知机的网络结构
        #-------------------fc-1--------------------
        output_fc1 = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=FLAGS.num_neurons, scope='fc-1')
        #-------------------fc-2--------------------
        output_fc2 = tf.contrib.layers.fully_connected(inputs=output_fc1, num_outputs=FLAGS.num_neurons, scope='fc-2')
        #-----------------dropout layer-------------
        output_dp = tf.contrib.layers.dropout(inputs=output_fc2, keep_prob=keep_prob, scope='dropout-layer')
        #-------------------fc-3--------------------
        output_pre_softmax = tf.contrib.layers.fully_connected(inputs=output_dp, num_outputs=FLAGS.num_class, scope='fc-3')

        # 定义 loss
        with tf.name_scope('loss'):
            loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_pre_softmax,
                                                                                labels=label_place, name='loss_tensor'))

        # 定义 accuracy
        with tf.name_scope('accuracy'):
            prediction = tf.equal(tf.arg_max(output_pre_softmax, 1), tf.arg_max(label_place, 1))
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        # 根据初值定义optimizer：(两种定义方式，第二种可以得到所有的 gradients 和 variables)
        # with tf.name_scope('train'):
        #     train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_tensor, global_step=global_step)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gradients_vars = optimizer.compute_gradients(loss_tensor)
            train_op = optimizer.apply_gradients(gradients_vars, global_step=global_step)
        
        # 定义 summaries
        tf.summary.scalar('loss', loss_tensor, collections=['train', 'test'])
        tf.summary.scalar('accuracy', accuracy, collections=['train', 'test'])
        tf.summary.scalar('global_step', global_step, collections=['train'])
        tf.summary.scalar('learning_rate', learning_rate, collections=['train'])

        # 按照 'train' 和 'test' 将两种不同类型的 summaries 分别 merge
        summary_train = tf.summary.merge_all('train')
        summary_test = tf.summary.merge_all('test')
        
        max_acc = 99.0 # 高于此精度的模型将被saved
        min_cross = 0.1
        # if not os.path.exists(FLAGS.checkpoints_path):
        #     os.makedirs(FLAGS.checkpoints_path)

        # 建立会话
        sess = tf.Session(graph=graph)
        with sess.as_default():
            # 定义 saver，初始化以及定义 train 和 test 的 summary writer
            saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoints)
            sess.run(tf.global_variables_initializer())

            train_summary_dir = os.path.join(FLAGS.events_path, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir)
            train_summary_writer.add_graph(sess.graph)

            test_summary_dir = os.path.join(FLAGS.events_path, 'summaries', 'test')
            test_summary_writer = tf.summary.FileWriter(test_summary_dir)
            test_summary_writer.add_graph(sess.graph)

            # 随机梯度下降进行训练
            for num_epoch in range(FLAGS.num_epochs):

                image_batch, label_batch = mnist.train.next_batch(FLAGS.batch_size)
                batch_loss, batch_acc, _, batch_summ, num_train, lr = sess.run([loss_tensor, accuracy, train_op, summary_train, global_step, learning_rate],
                                                                            feed_dict={image_place: image_batch, label_place: label_batch,
                                                                                        keep_prob: FLAGS.keep_prob})
                batch_acc *= 100
                progress = float(num_epoch % FLAGS.num_batch + 1) / FLAGS.num_batch
                num_epoch_batch = num_epoch // FLAGS.num_batch + 1
                print_progress(progress, num_epoch_batch, batch_loss, batch_acc)
                # print('Epoch '+str(num_epoch+1) + ', learning rate is %.4f' % lr + ', train accuracy is '+'{:.2f}%.'.format(batch_acc))
                # print('Epoch '+str(num_epoch+1) + ', learning rate is %.4f' % lr + ', train loss is '+ '{:.4f}, '.format(batch_loss) + \
                #         'accuracy is '+'{:.2f}%.'.format(batch_acc))
                train_summary_writer.add_summary(batch_summ, num_epoch)

                checkpoints_prefix = 'model.ckpt'
                if (num_epoch+1) % FLAGS.num_batch == 0:
                    print('Learning rate is %.4f' % lr + ', train accuracy is '+'{:.2f}%.'.format(batch_acc))
                    if (batch_loss <= min_cross) & (batch_acc > max_acc): # 按照要求保存网络模型
                        min_cross = batch_loss
                        max_acc = batch_acc
                        saver.save(sess, os.path.join(FLAGS.checkpoints_path, checkpoints_prefix), global_step=num_epoch+1)
                        # print("\033[0;31;40m\tModel restored ... ...\033[0m\n")
                        printRed("Model restored ... ...\n")
                        print('\n')

                    if FLAGS.online_test: #定义是否实时显示测试集精度
                        test_loss, test_acc, test_summ = sess.run([loss_tensor, accuracy, summary_test],
                                                                    feed_dict={image_place: test_image, label_place: test_label,
                                                                    keep_prob: 1.})
                        test_acc *= 100
                        print('Test accuracy is '+'{:.2f}%.\n'.format(test_acc))
                        test_summary_writer.add_summary(test_summ, num_epoch)
            
            # 测试集精度
            test_accuracy = sess.run([accuracy], feed_dict={image_place: test_image, label_place: test_label, 
                                                            keep_prob: 1.})
            test_acc = test_accuracy[0]*100
            print("Final Test Accuracy is %.2f%%" % test_acc)

            train_summary_writer.close()
            test_summary_writer.close()
        sess.close()

# 关于 restore 模型的代码详见 Logistic_regression.py 最后

# 该程序通过 TensorFlow 搭建4层卷积神经网络实现对人脸数据集 FaceWarehouse 的分类
# 关于人脸库 FaceWarehouse，详见：http://kunzhou.net/zjugaps/facewarehouse/
# 训练集、测试集图像与标签的数据存储于 data_path中的faces_150.mat
# 程序采用批量归一化(batch normalization)解决协变量偏移问题
# -*- coding: utf-8 -*- 
"""
Created on 19 May, 2019
@author jswanglp

requirements:
    Keras==2.2.4
    matplotlib==2.0.2
    numpy==1.15.4
    tensorflow==1.12.0
    scipy==1.1.0
    Bunch==1.0.1

"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io as scio
import os
from keras.utils import to_categorical
from Bunch import *
from progress_bar import print_progress

tf.logging.set_verbosity(tf.logging.ERROR)

# 定义初始化、卷积、池化函数
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':

    tf.app.flags.DEFINE_integer('num_epochs', 400, 'The number of epoch, default is 400.')
    tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size, default is 128.')
    tf.app.flags.DEFINE_float('decay_steps', 200., 'Decay steps of learning rate, default is 200.')
    tf.app.flags.DEFINE_float('keep_prob', 0.5, 'Keep_prob, default is 0.8')
    tf.app.flags.DEFINE_boolean('online_test', True, 'Online test or not, default is True.')
    FLAGS = tf.app.flags.FLAGS

    data_path = 'GoogleDrive/My Drive/MATLAB/face recognition/faces_database/faces_150.mat'
    # data_path = 'GoogleDrive/My Drive/MATLAB/face recognition/faces_database/faces_150_equalhis.mat'

    events_path = os.path.dirname(os.path.abspath(__file__)) + '/Tensorboard'
    checkpoints_path = os.path.dirname(os.path.abspath(__file__)) + '/Checkpoints'
    
    data = scio.loadmat(data_path)
    # -------------------训练集图像提取---------------------------------------------
    train_image = data['train_faces']
    train_labels = to_categorical(data['train_labels'].flatten(), num_classes=150)
    train_data = Bunch(train_image=train_image, train_labels=train_labels)
    print('\n', 'Train image set extraction completed... ...\n')

    # ----------------测试集图像提取------------------------------------------------
    test_image = data['test_faces']
    test_labels = to_categorical(data['test_labels'].flatten(), num_classes=150)
    test_data = Bunch(test_image=test_image, test_labels=test_labels)
    print(' Test image set extraction completed... ...\n')

    model_name = 'model.ckpt'
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # decay_steps = 100
        decay_rate = 0.8
        start_rate = 1e-3
        learning_rate = tf.train.exponential_decay(start_rate,
                                                    global_step=global_step,
                                                    decay_steps=FLAGS.decay_steps,
                                                    decay_rate=decay_rate,
                                                    staircase=True,
                                                    name='exponential_decay')
        
        with tf.name_scope('Input'):
            x = tf.placeholder("float", shape=[None, 90, 75])
            y = tf.placeholder("float", shape=[None, 150])
            keep_prob = tf.placeholder("float") # keep_prob表示每个神经元被激活的概率
            is_training = tf.placeholder(tf.bool, [])
        with tf.name_scope('Input_images'):
            x_image = tf.reshape(x, [-1, 90, 75, 1])

    # --------------conv1-----------------------------------45*38*32
        with tf.name_scope('Conv1'):
            with tf.name_scope('weights_conv1'):
                W_conv1 = weight_variable([3, 3, 1, 32], name='w_conv1')
            with tf.name_scope('bias_covn1'):
                b_conv1 = bias_variable([32], name='b_conv1')
    
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            with tf.name_scope('features_conv1'):
                h_pool1 = max_pool_2x2(h_conv1)
    # --------------conv2-----------------------------------23*19*64
        with tf.name_scope('Conv2'):
            with tf.name_scope('weights_conv2'):
                W_conv2 = weight_variable([3, 3, 32, 64], name='w_conv2')
            with tf.name_scope('bias_covn2'):
                b_conv2 = bias_variable([64], name='b_conv2')
    
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            with tf.name_scope('features_conv2'):
                h_pool2 = max_pool_2x2(h_conv2)
    # --------------conv3-----------------------------------12*10*128
        with tf.name_scope('Conv3'):
            W_conv3 = weight_variable([3, 3, 64, 128], name='w_conv3')
            b_conv3 = bias_variable([128], name='b_conv3')

            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)
    # --------------conv4-----------------------------------6*5*256
        with tf.name_scope('Conv4'):
            W_conv4 = weight_variable([3, 3, 128, 256], name='w_conv4')
            b_conv4 = bias_variable([256], name='b_conv4')

            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)
    # --------------conv5-----------------------------------3*3*512
        with tf.name_scope('Conv5'):
            W_conv5 = weight_variable([3, 3, 256, 512], name='w_conv5')
            b_conv5 = bias_variable([512], name='b_conv5')

            h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
            h_pool5 = max_pool_2x2(h_conv5)

    # --------------fc--------------------------------------
        with tf.name_scope('FC1'):
            h_pool5_flat = tf.layers.flatten(h_pool5, name='pool5_flatten')
            num_f = h_pool5_flat.get_shape().as_list()[-1]
            W_fc1 = weight_variable([num_f, 1024], name='w_fc1')
            b_fc1 = bias_variable([1024], name='b_fc1')
            # h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1) # y=wx+b或者y.T=(x.T)(w.T)+b.T，其中y为列向量
            h_bn6 = tf.layers.batch_normalization(tf.matmul(h_pool5_flat, W_fc1) + b_fc1, training=is_training)
            h_fc1 = tf.nn.relu(h_bn6)
        with tf.name_scope('Dropout1'):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        with tf.name_scope('FC2'):
            W_fc2 = weight_variable([1024, 1024], name='w_fc2')
            b_fc2 = bias_variable([1024], name='b_fc2')
            # h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            h_bn7 = tf.layers.batch_normalization(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, training=is_training)
            h_fc2 = tf.nn.relu(h_bn7)
        with tf.name_scope('Dropout2'):
            h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        with tf.name_scope('OutPut_layer'):
            W_fc3 = weight_variable([1024, 150], name='w_fc2')
            b_fc3 = bias_variable([150], name='b_fc2')
            y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
            
    # ---------------------loss-----------------------------
        with tf.name_scope('Loss'):
            # y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            # cross_entropy = -tf.reduce_mean(y * tf.log(y_conv + 1e-10)) # 防止log0
        # or like
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                            logits=y_conv))

        with tf.name_scope('Train'):
            # 有 bn 层需要添加 update_ops 至 train_step
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
            train_step = tf.group([train_step, update_ops])

        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
        max_acc = 101.0
        min_cross = 0.1
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
            sess.run(tf.global_variables_initializer())
    
            print('Training ========== (。・`ω´・) ========')
            for epoch_num in range(FLAGS.num_epochs):
                train_s = np.c_[train_data.train_image.reshape((1500,-1)), train_data.train_labels]
                np.random.shuffle(train_s)
                max_size = train_s.shape[0] // FLAGS.batch_size
                for num in range(max_size):
                    batch = [train_s[num*FLAGS.batch_size:(num+1)*FLAGS.batch_size, :90*75].reshape((-1,90,75)), 
                            train_s[num*FLAGS.batch_size:(num+1)*FLAGS.batch_size, -150:]]
                    _, acc, loss = sess.run([train_step, accuracy, cross_entropy], 
                                            feed_dict={x: batch[0], y: batch[1], 
                                            keep_prob: FLAGS.keep_prob, 
                                            is_training: True})

                    acc *= 100
                    num_iter = max_size * 10
                    progress = ((epoch_num * max_size + num) % num_iter + 1) / num_iter
                    num_ep = epoch_num + 1
                    print_progress(progress, num_ep, loss, acc)

                if FLAGS.online_test and (epoch_num + 1) % 10 ==0 :
                    print(' '*12, 'Online-Testing ========== (。・`ω´・) ========')
                    imgs_t, labels_t = test_data.test_image.reshape((-1, 90, 75)), test_data.test_labels
                    test_acc, test_loss = sess.run([accuracy, cross_entropy], feed_dict={x: imgs_t, y: labels_t,
                                                                                        keep_prob: 1.0, 
                                                                                        is_training: False})
                    test_acc *= 100
                    print(' '*10, 'Loss on testing data is %.4f, accuracy is %.2f%%.' %(test_loss, test_acc))
                    print('\nKeep on training ========== (。・`ω´・) ========')
        
                # 保存精度高的3个模型(以acc和loss判断)
                if (loss <= min_cross) & (acc >= max_acc) & (epoch_num > 100): # step比tensorboard中数大1
                    min_cross = loss
                    max_acc = acc
                    saver.save(sess, os.path.join(checkpoints_path, model_name), global_step=epoch_num)

            test_im, test_lab = train_data.train_image[0].reshape((-1, 90, 75)), train_data.train_labels[0].reshape((-1, 150))
            feature_map1 = sess.run(h_pool1, feed_dict={x: test_im, y: test_lab, keep_prob: 1.0, is_training: False})
            # feature_map2 = sess.run(h_pool2, feed_dict={x: test_im, y: test_lab, keep_prob: 1.0, is_training: False})
    
        sess.close()
    print('\n', 'Training completed.')
    
    # with tf.Session() as sess: # 开启会话，复原保存的网络
        # model_path = 'Tensorboard/f_map.ckpt-241'
        # saver.restore(sess, model_path)
        # acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: test_data.test_image, 
                                # y: test_data.test_labels, keep_prob: 1.0})
        # acc_p = acc*100
        # print('Accuracy is %.2f' %(acc_p), '%.')
    # sess.close()
    
# ----------------显示train_data.image第一幅图像的第一层 feature map (45*38*32)------------

    f_map = feature_map1.reshape((45, 38, 32))
    num_map = range(f_map.shape[-1])
    fig = plt.figure(1,figsize=(24, 14))
    G = gridspec.GridSpec(4, 8)
    G.hspace,G.wspace = .05,.05
    try:
        for i in range(4):
            for j in range(8):
                plt.subplot(G[i, j])
                num = i * 5 + j
                plt.imshow(f_map[:, :, num], 'gray')
                plt.xticks([]), plt.yticks([])
    except IndexError:
        plt.xticks([]), plt.yticks([])
    plt.show()

# Программой является простое пособие по tensorboard
# -*- coding: utf-8 -*- 
"""
Created on 15 May, 2019
@author jswanglp

requirements:
    numpy==1.15.4
    matplotlib==2.0.2
    tensorflow==1.12.0
    tensorflow_datasets==1.0.1
    utils==1.0.1

"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time,os

if __name__ == '__main__':

    # Считывание обучающих, тестовых изображений от CIFAR-10
    cifar_train = tfds.as_numpy(tfds.load("cifar10", split=tfds.Split.TRAIN, batch_size=-1))
    imgs_train, labels_train = cifar_train['image'].reshape(-1, 3072) / 255., cifar_train['label']

    cifar_test = tfds.as_numpy(tfds.load("cifar10", split=tfds.Split.TEST, batch_size=-1))
    imgs_test, labels_test = cifar_test['image'].reshape(-1, 3072) / 255., cifar_test['label']
    
    # Настройка параметров
    learning_rate = 5e-4 #@param {type:"number"}
    batch_size = 196 #@param {type:"integer"}
    num_epochs = 100 #@param {type:"integer"}

    event_path = './Tensorboard'
    checkpoints_path = './Checkpoints'

    # Структура graph
    graph = tf.Graph()
    with graph.as_default():

        global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.name_scope('Input'):
            x = tf.placeholder(tf.float32, shape=[None, 3072], name='input_images')
            x_imgs = tf.reshape(x, shape=[-1, 32, 32, 3], name='images')
            y_p = tf.placeholder(tf.int32, shape=[None, ], name='labels')
            y = tf.one_hot(y_p, depth=10, name='one_hot_labels')
            keep_pro = tf.placeholder(tf.float32)

        with tf.name_scope('Conv_layers'):
            with tf.name_scope('Conv1'):
                w_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], stddev=0.1), name='weights_conv1')
                b_1 = tf.Variable(tf.constant(0.1, shape=[64]), name='bias_conv1')
                h_conv1 = tf.nn.relu(tf.nn.conv2d(x_imgs, w_1, strides=[1, 1, 1, 1], padding='SAME') + b_1)
                h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            with tf.name_scope('Conv2'):
                w_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1), name='weights_conv2')
                b_2 = tf.Variable(tf.constant(0.1, shape=[128]), name='bias_conv2')
                h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_2, strides=[1, 1, 1, 1], padding='SAME') + b_2)
                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            with tf.name_scope('Conv3'):
                w_3 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1), name='weights_conv3')
                b_3 = tf.Variable(tf.constant(0.1, shape=[256]), name='bias_conv3')
                h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_3, strides=[1, 1, 1, 1], padding='SAME') + b_3)
                h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        with tf.name_scope('Fc_layers'):
            with tf.name_scope('Fc1'):
                h_pool3_fla = tf.layers.flatten(h_pool3)
                num_f = h_pool3_fla.get_shape().as_list()[-1]

                w_fc1 = tf.Variable(tf.truncated_normal([num_f, 512], stddev=0.1), name='weights_fc1')
                b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name='bias_fc1')
                h_fc1 = tf.nn.relu(tf.matmul(h_pool3_fla, w_fc1) + b_fc1)
                h_drop1 = tf.nn.dropout(h_fc1, keep_prob=keep_pro, name='Dropout1')

            with tf.name_scope('Fc2'):
                w_fc2 = tf.Variable(tf.truncated_normal([512, 256], stddev=0.1), name='weights_fc2')
                b_fc2 = tf.Variable(tf.constant(0.1, shape=[256]), name='bias_fc2')
                h_fc2 = tf.nn.relu(tf.matmul(h_drop1, w_fc2) + b_fc2)
                h_drop2 = tf.nn.dropout(h_fc2, keep_prob=keep_pro, name='Dropout2')

        with tf.name_scope('Output'):
            w_op = tf.Variable(tf.truncated_normal([256, 10], stddev=0.1), name='weights_op')
            b_op = tf.Variable(tf.constant(0.1, shape=[10]), name='bias_op')
            h_op = tf.matmul(h_drop2, w_op) + b_op
        
        # L2 регуляризация
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_fc1)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w_fc2)
        regularizer = tf.contrib.layers.l2_regularizer(scale=150./50000)
        reg_tem = tf.contrib.layers.apply_regularization(regularizer)

        with tf.name_scope('loss'):
            # entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_op))
            entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=h_op) + reg_tem)
        
        with tf.name_scope('accuracy'):
            prediction = tf.cast(tf.equal(tf.arg_max(h_op, 1), tf.argmax(y, 1)), "float")
            accuracy = tf.reduce_mean(prediction)
        
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(entropy_loss, global_step=global_step)
        
        # Summaries
        tf.summary.image('input_images', x_imgs, max_outputs=3, collections=['train', 'test'])
        tf.summary.histogram('conv1_weights', w_1, collections=['train'])
        tf.summary.histogram('conv1_bias', b_1, collections=['train'])
        tf.summary.scalar('loss', entropy_loss, collections=['train', 'test'])
        tf.summary.scalar('accuracy', accuracy, collections=['train', 'test'])

        summ_train = tf.summary.merge_all('train')
        summ_test = tf.summary.merge_all('test')

    # Обучение модели
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        summ_train_dir = os.path.join(event_path, 'summaries','train')
        summ_train_Writer = tf.summary.FileWriter(summ_train_dir)
        summ_train_Writer.add_graph(sess.graph)

        summ_test_dir = os.path.join(event_path, 'summaries', 'test')
        summ_test_Writer = tf.summary.FileWriter(summ_test_dir)
        summ_test_Writer.add_graph(sess.graph)

        for num in range(num_epochs):
            imgs_data = np.c_[imgs_train, labels_train]
            np.random.shuffle(imgs_data)
            num_batchs = imgs_train.shape[0] // batch_size
            start = time.time()
            for num_ep in range(num_batchs):
                # start = time.time()
                imgs_batch = imgs_data[num_ep*batch_size:(num_ep+1)*batch_size, :-1]
                labels_batch = imgs_data[num_ep*batch_size:(num_ep+1)*batch_size,-1]
                _, acc, loss, rt, num_step = sess.run([train_op, accuracy, entropy_loss, summ_train, global_step], 
                                                        feed_dict={x: imgs_batch, y_p: labels_batch, keep_pro: 0.5})
            summ_train_Writer.add_summary(rt, global_step=num_step)
            end = time.time()
            acc *= 100
            num_e = str(num + 1)
            print_list = [num_e, loss, acc]
            print("Epoch {0[0]}, train_loss is {0[1]:.4f}, accuracy is {0[2]:.2f}%.".format(print_list))
            print("Running time is {0:.2f}s.".format(end-start))
            _, acc, loss, rs = sess.run([train_op, accuracy, entropy_loss, summ_test], feed_dict={x: imgs_test,
                                                                                        y_p: labels_test,
                                                                                        keep_pro: 1.})
            summ_test_Writer.add_summary(rs, global_step=num_step)
            acc *= 100
            print_list = [loss, acc]
            print("Test_loss is {0[0]:.4f}, accuracy is {0[1]:.2f}%.\n".format(print_list))
        print('Training completed.')
    # sess.close()
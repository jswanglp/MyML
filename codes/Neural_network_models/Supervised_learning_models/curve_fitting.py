# 该程序通过含一个隐层的神经网络对输入数据进行拟合
# 可以与岭回归的结果做比较
# -*- coding: utf-8 -*- 
"""
Created on 05 June, 2019
@author jswanglp

requirements:
    Keras==2.2.4
    matplotlib==2.0.2
    numpy==1.15.4
    tensorflow==1.12.0
    scipy==1.1.0
    Bunch==1.0.1

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

    
if __name__ == '__main__':
    
    num_epoch = 200
    # 样本数据的预处理
    data = np.array([[-2.95507616, 10.94533252],
                    [-0.44226119, 2.96705822],
                    [-2.13294087, 6.57336839],
                    [1.84990823, 5.44244467],
                    [0.35139795, 2.83533936],
                    [-1.77443098, 5.6800407],
                    [-1.8657203, 6.34470814],
                    [1.61526823, 4.77833358],
                    [-2.38043687, 8.51887713],
                    [-1.40513866, 4.18262786]])
    x = data[:, 0]
    y = data[:, 1]
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    # 预测数据数量多于初始数据样本数
    x_pre = np.linspace(x.min(), x.max(), 30, endpoint=True).reshape(-1, 1)

    # 网络图设置
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('Input'):
            x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
            y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
        with tf.name_scope('FC'):
            w_1 = tf.get_variable('w_fc1', shape=[1, 32], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_1 = tf.get_variable('b_fc1', initializer=tf.constant(0.1, shape=[32]))
            layer_1 = tf.nn.sigmoid(tf.matmul(x, w_1) + b_1)
        with tf.name_scope('Output'):
            w_2 = tf.get_variable('w_fc2', shape=[32, 1], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_2 = tf.get_variable('b_fc2', initializer=tf.constant(0.1, shape=[1]))
            layer_2 = tf.matmul(layer_1, w_2) + b_2
        
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.pow(layer_2 - y, 2))
        with tf.name_scope('Train'):
            train_op = tf.train.AdamOptimizer(learning_rate=3e-1).minimize(loss)

    # 模型训练
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        time_start = time.time()
        for num in range(num_epoch):
            _, ls = sess.run([train_op, loss], feed_dict={x: X, y: Y})
            print_list = [num+1, ls]
            if (num+1) % 10 == 0 or num == 0:
                print('Epoch {0[0]}, loss: {0[1]:.4f}.'.format(print_list))
        
        # time_start = time.time()
        y_pre = sess.run(layer_2, feed_dict={x: x_pre})
        sess.close()
        time_end = time.time()
        t = time_end - time_start
        print('Running time is: %.4f' % t)
    
    # 预测曲线
    data_pre = np.c_[x_pre, y_pre]
    DATA = [data, data_pre]
    NAME = ['Training data', 'Fitting curve']
    STYLE = ['*r', 'b']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
    for dat, name, style in zip(DATA, NAME, STYLE):
        ax.plot(dat[:, 0], dat[:, 1], style, markersize=8, label=name)
        ax.legend(loc='upper right', fontsize=14)
        ax.tick_params(labelsize=14)
    plt.show()

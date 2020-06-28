# 该程序通过 tensorflow 的 tensor_forest API实现随机森林对随机数据的分类
# 参考程序：https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/random_forest.ipynb
# -*- coding: utf-8 -*- 
"""
Created on 16 May, 2019
@author jswanglp

requirements:
    numpy==1.15.4
    tensorflow==1.12.0
    scipy==1.1.0
    hmmlearn==0.2.3
    matplotlib==2.0.2
    graphviz==0.14
    scikit_learn==0.23.1

"""

import tensorflow as tf
from tensorflow.python.ops import resources
from tensorflow.contrib.tensor_forest.python import tensor_forest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 绑定数据
class Bunch(dict):  
	def __init__(self,*args,**kwds):  
		super(Bunch,self).__init__(*args,**kwds)  
		self.__dict__ = self

# 生成带标签的随机数据
def generate_random(sigma, N, mu1=[15., 25., 10], mu2=[30., 40., 30], mu3=[25., 10., 20], mu4=[40., 30., 40]):  
	c = sigma.shape[-1]        #生成N行c维的随机测试数据
	X = np.zeros((N, c))       # 初始化X，2行N列。2维数据，N个样本 
	target = np.zeros((N,1))
	for i in range(N):  
		if np.random.random(1) < 0.25:  # 生成0-1之间随机数  
			X[i, :]  = np.random.multivariate_normal(mu1, sigma[0, :, :], 1)     #用第一个高斯模型生成2维数据  
			target[i] = 0
		elif 0.25 <= np.random.random(1) < 0.5:  
			X[i, :] = np.random.multivariate_normal(mu2, sigma[1, :, :], 1)      #用第二个高斯模型生成2维数据  
			target[i] = 1
		elif 0.5 <= np.random.random(1) < 0.75:  
			X[i, :] = np.random.multivariate_normal(mu3, sigma[2, :, :], 1)      #用第三个高斯模型生成2维数据  
			target[i] = 2
		else:  
			X[i, :] = np.random.multivariate_normal(mu4, sigma[3, :, :], 1)      #用第四个高斯模型生成2维数据  
			target[i] = 3
	return X, target

if __name__ == '__main__':

    # 生成训练、测试集数据
    k, N = 4, 400
    sigma = np.zeros((k, 3, 3))
    for i in range(k):
    	sigma[i, :, :] = np.diag(np.random.randint(10, 25, size=(3, )))
    sample, target = generate_random(sigma, N)
    feature_names = ['x_label', 'y_label', 'z_label'] # 特征数
    target_names = ['gaussian1', 'gaussian2', 'gaussian3', 'gaussian4'] # 类别
    data = Bunch(sample=sample, feature_names=feature_names, target=target, target_names=target_names)
    sample_t, target_t = generate_random(sigma, N)
    data_t = Bunch(sample=sample_t, target=target_t)

    # 模型参数设置
    num_steps = 20
    batch_size = 256
    num_classes = 4
    num_features = 3
    num_trees = 5
    max_nodes = 20

    X = tf.placeholder(tf.float32, shape=[None, num_features])
    Y = tf.placeholder(tf.int32, shape=[None])

    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                          num_features=num_features,
                                          num_trees=num_trees,
                                          max_nodes=max_nodes).fill()

    # RF 的图模型
    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    train_op = forest_graph.training_graph(X, Y)
    loss_op = forest_graph.training_loss(X, Y)

    # 正确率
    infer_op, _, _ = forest_graph.inference_graph(X)
    correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化
    init_vars = tf.group(tf.global_variables_initializer(),
        resources.initialize_resources(resources.shared_resources()))
    
    # 设置会话
    sess = tf.train.MonitoredSession()

    sess.run(init_vars)

    # 训练模型
    for i in range(1, num_steps + 1):
        batch_data = np.c_[data.sample, data.target]
        np.random.shuffle(batch_data)
        batch_x, batch_y = batch_data[:batch_size, :-1], batch_data[:batch_size, -1]
        _, l, acc = sess.run([train_op, loss_op, accuracy_op], feed_dict={X: batch_x, Y: batch_y})
        acc *= 100
        print_list = [i, l, acc]
        print('Step {0[0]}, loss: {0[1]:.4f}, accuracy: {0[2]:.2f}%.'.format(print_list))
    # 训练集
    acc, pre_train = sess.run([accuracy_op, infer_op], feed_dict={X: data.sample, Y: data.target.flatten()})
    acc *= 100
    print('Accuracy on training set: %.2f.' % acc)
    # 测试集
    test_x, test_y = data_t.sample, data_t.target.flatten()
    acc_t, pre_test = sess.run([accuracy_op, infer_op], feed_dict={X: test_x, Y: test_y})
    acc_t *= 100
    print('Accuracy on testing set: %.2f.' % acc_t)

    sess.close()

    # 显示训练，测试数据的分布
    target_train = data.target.flatten().astype('int32')
    target_pre = np.argmax(pre_train, axis=1).astype('int32')
    titles = ['Random training data', 'Classified training data by RF']
    TAR = [target_train, target_pre]
    DATA = [data.sample, data.sample]
    fig = plt.figure(1, figsize=(16, 8))
    fig.subplots_adjust(wspace=.01, hspace=.02)
    for i, title, data_n, tar in zip([1, 2], titles, DATA, TAR):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        if title == 'Random training data':
            ax.scatter(data_n[:,0], data_n[:,1], data_n[:,2], c='b', s=35, alpha=0.4, marker='o')
        else:
            color=['b','r','g','y']
            for j in range(N):
                ax.scatter(data_n[j, 0], data_n[j, 1], data_n[j, 2], c=color[tar[j]], s=35, alpha=0.4, marker='P')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20., azim=-25)
        ax.set_title(title, fontsize=14, y=0.01)
    # plt.show()

    # 显示 RF 对测试数据的分类情况
    target_test = data_t.target.flatten().astype('int32')
    target_pre = np.argmax(pre_test, axis=1).astype('int32')
    titles = ['Random testing data', 'Classified testing data by RF']
    TAR = [target_test, target_pre]
    DATA = [data_t.sample, data_t.sample]
    fig = plt.figure(2, figsize=(16, 8))
    fig.subplots_adjust(wspace=.01, hspace=.02)
    for i, title, data_n, tar in zip([1, 2], titles, DATA, TAR):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        if title == 'Random testing data':
            ax.scatter(data_n[:,0], data_n[:,1], data_n[:,2], c='b', s=35, alpha=0.4, marker='o')
        else:
            color=['b','r','g','y']
            for j in range(N):
                ax.scatter(data_n[j, 0], data_n[j, 1], data_n[j, 2], c=color[tar[j]], s=35, alpha=0.4, marker='P')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20., azim=-25)
        ax.set_title(title, fontsize=14, y=0.01)
    plt.show()

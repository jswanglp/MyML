# 该程序通过调用 hmmlearn 的API GaussianHMM 实现随机数据的聚类
# 也可以用于分类，区别在于：
# 聚类时全部样本用于一个模型，分类时每个类拥有自己独立的模型
# hmmlearn API Reference: https://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn-hmm
# Hidden Markov Model
# coding: utf-8
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bunch import *

# 生成带标签的随机数据
def generate_random(sigma, N, mu1=[15., 25., 10], mu2=[30., 40., 30], mu3=[25., 10., 20], mu4=[40., 30., 40]):  
	c = sigma.shape[-1]        #生成N行c维的随机测试数据
	X = np.zeros((N, c))       # 初始化X，N个样本 
	target = np.zeros((N,1))
	for i in range(N):  
		if np.random.random(1) < 0.25:  # 生成0-1之间随机数  
			X[i, :]  = np.random.multivariate_normal(mu1, sigma[0, :, :], 1)     # 用第一个高斯模型生成3维数据  
			target[i] = 0
		elif 0.25 <= np.random.random(1) < 0.5:  
			X[i, :] = np.random.multivariate_normal(mu2, sigma[1, :, :], 1)      # 用第二个高斯模型生成3维数据  
			target[i] = 1
		elif 0.5 <= np.random.random(1) < 0.75:  
			X[i, :] = np.random.multivariate_normal(mu3, sigma[2, :, :], 1)      # 用第三个高斯模型生成3维数据  
			target[i] = 2
		else:  
			X[i, :] = np.random.multivariate_normal(mu4, sigma[3, :, :], 1)      # 用第四个高斯模型生成3维数据  
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

    # 训练模型
    model = hmm.GaussianHMM(n_components=4,covariance_type='tied')
    model.fit(data.sample)

    target_train = data.target.flatten().astype('int32')
    target_pre = model.predict(data.sample).astype('int32')
    # 显示 HMM 对训练数据的聚类结果
    titles = ['Random training data', 'Clustering training data by HMM']
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

    target_test = data_t.target.flatten().astype('int32')
    target_pre = model.predict(data_t.sample).astype('int32')
    # 显示 HMM 对测试数据的聚类结果
    titles = ['Random testing data', 'Clustering testing data by HMM']
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

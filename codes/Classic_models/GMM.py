# 该程序实现EM算法对混合高斯模型参数的估计
# 混合高斯模型对随机数据的聚类
# coding: utf-8
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bunch import *
from GaussianMM import *

# 生成带标签的随机数据
def generate_random(sigma, N, mu1=[15., 25., 10], mu2=[30., 40., 30], mu3=[25., 10., 20], mu4=[40., 30., 40]):
    c = sigma.shape[-1]        # 生成N行c维的随机测试数据，比较kmeans与decision tree
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

    # 生成带标签的随机数据
    k, N = 4, 400
    # 初始化方差，生成样本与标签
    sigma = np.zeros((k, 3, 3))
    for i in range(k):
        sigma[i, :, :] = np.diag(np.random.randint(10, 25, size=(3, )))
    sample, target = generate_random(sigma, N)
    feature_names = ['x_label', 'y_label', 'z_label'] # 特征数
    target_names = ['gaussian1', 'gaussian2', 'gaussian3', 'gaussian4'] # 类别
    data = Bunch(sample=sample, feature_names=feature_names, target=target, target_names=target_names)

    # 初始化模型参数
    model = GaussianMM()
    err_mu = 1e-4
    err_alpha = 1e-4
    # -------------二类----------------
    model.init_fn(f_dim=3, num_mixed=2)
    # print('mu:\n', model.mu)
    # print('sigma:\n', model.sigma)
    # print('alpha:\n', model.alpha)
    # 迭代训练，直到满足收敛条件
    model.fit(data.sample, err_mu=err_mu, err_alpha=err_alpha, max_iter=100)
    # 预测每个样本属于哪个成分
    tar2 = model.predict(data.sample)
    # -------------三类----------------
    model.init_fn(f_dim=3, num_mixed=3)
    model.fit(data.sample, err_mu=err_mu, err_alpha=err_alpha, max_iter=100)
    tar3 = model.predict(data.sample)
    # -------------四类----------------
    model.init_fn(f_dim=3, num_mixed=4)
    model.fit(data.sample, err_mu=err_mu, err_alpha=err_alpha, max_iter=100)
    tar4 = model.predict(data.sample)

    # 显示训练数据的分布以及聚类结果
    # 训练数据与二类
    titles = ['Random training data', 'Clustered data by 2-GMM']
    DATA = [data.sample, data.sample]
    color=['b','r','g','y']
    fig = plt.figure(1, figsize=(16, 8))
    fig.subplots_adjust(wspace=.01, hspace=.02)
    for i, title, data_n in zip([1, 2], titles, DATA):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        if title == 'Random training data':
            ax.scatter(data_n[:,0], data_n[:,1], data_n[:,2], c='b', s=35, alpha=0.4, marker='o')
        else:
            for j in range(N):
                ax.scatter(data_n[j, 0], data_n[j, 1], data_n[j, 2], c=color[tar2[j]], s=35, alpha=0.4, marker='P')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20., azim=-25)
        ax.set_title(title, fontsize=14)
    
    # 三类与四类
    titles = ['Clustered data by 3-GMM', 'Clustered data by 4-GMM']
    TAR = [tar3, tar4]
    fig = plt.figure(2, figsize=(16, 8))
    fig.subplots_adjust(wspace=.01, hspace=.02)
    for i, title, data_n, tar in zip([1, 2], titles, DATA, TAR):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        for j in range(N):
            ax.scatter(data_n[j, 0], data_n[j, 1], data_n[j, 2], c=color[tar[j]], s=35, alpha=0.4, marker='P')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20., azim=-25)
        ax.set_title(title, fontsize=14)
    plt.show()
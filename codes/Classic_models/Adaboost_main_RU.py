# В программе реализован алгоритм AdaBoost для классификации стохастических данных
# Оригинальная программа: https://github.com/wzyonggege/statistical-learning-method/blob/master/AdaBoost/Adaboost.ipynb
# -*- coding: utf-8 -*- 
"""
Created on 14 May, 2019
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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bunch import *
from Adaboost import *

# Генератор стохастических данных
def generate_random(sigma, N, mu1=[15., 25., 20], mu2=[30., 40., 30]):  
	c = sigma.shape[-1]
	X = np.zeros((N, c))
	target = np.zeros((N,1))
	for i in range(N):  
		if np.random.random(1) < 0.5:
			X[i, :]  = np.random.multivariate_normal(mu1, sigma[0, :, :], 1)     # первая гауссовская модель
			target[i] = 1
		else:  
			X[i, :] = np.random.multivariate_normal(mu2, sigma[1, :, :], 1)      # вторая гауссовская модель
			target[i] = -1
	return X, target

if __name__ == '__main__':

    # Стохастические данные
    k, N = 2, 400
    sigma = np.zeros((k, 3, 3))
    for i in range(k):
    	sigma[i, :, :] = np.diag(np.random.randint(10, 25, size=(3, )))
    sample, target = generate_random(sigma, N)
    feature_names = ['x_label', 'y_label', 'z_label']
    target_names = ['gaussian1', 'gaussian2', 'gaussian3', 'gaussian4']
    data = Bunch(sample=sample, feature_names=feature_names, target=target, target_names=target_names)
    sample_t, target_t = generate_random(sigma, N)
    data_t = Bunch(sample=sample_t, target=target_t)

    # Обучение модели
    model = AdaBoost(n_estimators=3, learning_rate=0.5)
    model.fit(data.sample, target)
    tar = [model.predict(x) for x in data.sample]
    tar_train = np.array([model.predict(x) for x in data.sample], dtype=np.int8) + 1
    tar_test = np.array([model.predict(x) for x in data_t.sample], dtype=np.int8) + 1
    acc_train = model.score(data.sample, data.target)
    acc_test = model.score(data_t.sample, data_t.target)
    print_list = [acc_train*100, acc_test*100]
    print('Accuracy on training set: {0[0]:.2f}%, accuracy on testing set: {0[1]:.2f}%.'.format(print_list))
    
    # Представление распределения обучающих и тестовых данных
    titles = ['Random training data', 'Random testing data']
    DATA = [data.sample, data_t.sample]
    fig = plt.figure(1, figsize=(16, 8))
    fig.subplots_adjust(wspace=.01, hspace=.02)
    for i, title, data_n in zip([1, 2], titles, DATA):
    	ax = fig.add_subplot(1, 2, i, projection='3d')
    	ax.scatter(data_n[:,0], data_n[:,1], data_n[:,2], c='b', s=35, alpha=0.4, marker='o')
    	ax.set_xlabel('X')
    	ax.set_ylabel('Y')
    	ax.set_zlabel('Z')
    	ax.view_init(elev=20., azim=-25)
    	ax.set_title(title, fontsize=14)
    
    # Представление результата классификации обучающих и тестовых данных
    titles = ['Classified training data by Adaboost', 'Classified testing data by Adaboost']
    TAR = [tar_train, tar_test]
    fig = plt.figure(2, figsize=(16, 8))
    fig.subplots_adjust(wspace=.01, hspace=.02)
    for i, title, data_n, tar in zip([1, 2], titles, DATA, TAR):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        color=['b','g', 'r']
        for j in range(N):
            ax.scatter(data_n[j, 0], data_n[j, 1], data_n[j, 2], c=color[tar[j]], s=35, alpha=0.4, marker='P')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20., azim=-25)
            ax.set_title(title, fontsize=14, y=0.01)
    plt.show()
# В программе реализован метод k-ближайших соседей для классификации трехмерных стохастических данных
# Оригинальная программа: https://github.com/wzyonggege/statistical-learning-method/blob/master/KNearestNeighbors/KNN.ipynb
# -*- coding: utf-8 -*- 
"""
Created on 12 May, 2019
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
from KNN import *

# Генератор стохастических данных
def generate_random(sigma, N, mu1=[25., 25., 20], mu2=[30., 40., 30]):  
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
    model = KNN(data.sample, target, n_neighbors=4, p=2)
    model.predict(data.sample[100])

    # Точность в данных тестирования
    acc = model.score(data_t.sample, data_t.target) * 100
    print('Accuracy on testing set: {:.2f}%.'.format(acc))
    tar_test = np.array([model.predict(x) for x in data_t.sample], dtype=np.int8) + 1

    # Представление результата классификации методом k-ближайших соседей
    titles = ['Random training data', 'Classified testing data by KNN']
    TAR = [target, tar_test]
    DATA = [data.sample, data_t.sample]
    fig = plt.figure(1, figsize=(16, 8))
    fig.subplots_adjust(wspace=.01, hspace=.02)
    for i, title, data_n, tar in zip([1, 2], titles, DATA, TAR):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        if title == 'Random training data':
            ax.scatter(data_n[:,0], data_n[:,1], data_n[:,2], c='b', s=35, alpha=0.4, marker='o')
        else:
            color=['b','g', 'r']
            for j in range(N):
                ax.scatter(data_n[j, 0], data_n[j, 1], data_n[j, 2], c=color[tar[j]], s=35, alpha=0.4, marker='P')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20., azim=-25)
        ax.set_title(title, fontsize=14, y=0.01)
    plt.show()
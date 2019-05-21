# В программе реализован наивный байесовский классификатор
# Оригинальная программа: https://github.com/wzyonggege/statistical-learning-method/blob/master/NaiveBayes/GaussianNB.ipynb
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from Bunch import *
from NaiveBayes import *

# Генератор стохастических данных
def generate_random(sigma, N, mu1=[15., 25., 10], mu2=[30., 40., 30], mu3=[25., 10., 20], mu4=[40., 30., 40]):  
    c = sigma.shape[-1]
    X = np.zeros((N, c))
    target = np.zeros((N,1))
    for i in range(N):  
        if np.random.random(1) < 0.25:
            X[i, :]  = np.random.multivariate_normal(mu1, sigma[0, :, :], 1)     # первая гауссовская модель
            target[i] = 0
        elif 0.25 <= np.random.random(1) < 0.5:  
            X[i, :] = np.random.multivariate_normal(mu2, sigma[1, :, :], 1)      # вторая гауссовская модель
            target[i] = 1
        elif 0.5 <= np.random.random(1) < 0.75:  
            X[i, :] = np.random.multivariate_normal(mu3, sigma[2, :, :], 1)      # третья гауссовская модель
            target[i] = 2
        else:  
            X[i, :] = np.random.multivariate_normal(mu4, sigma[3, :, :], 1)      # четвертая гауссовская модель
            target[i] = 3
    return X, target

if __name__ == '__main__':

    # Стохастические данные
    k, N = 4, 400
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
    model = NaiveBayes()
    model.fit(data.sample, data.target.flatten())
    tar_train = np.array([model.predict(x) for x in data.sample], dtype=np.uint8)
    tar_test = np.array([model.predict(x) for x in data_t.sample], dtype=np.uint8)
    acc_train = model.score(data.sample, data.target.flatten())
    acc_test = model.score(data_t.sample, data_t.target.flatten())
    print_list = [acc_train*100, acc_test*100]
    print('Accuracy on training set: {0[0]:.2f}%, accuracy on testing set: {0[1]:.2f}%.'.format(print_list))

    # Пример
    summary = model.normalized_prob(data_t.sample[100])
    print(summary)

    # Распределения обучающих и тестовых данных
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

    # Представление результата классификации методом наивного байесовского классификатора
    titles = ['Classified training data by NaiveBayes', 'Classified testing data by NaiveBayes']
    TAR = [tar_train, tar_test]
    fig = plt.figure(2, figsize=(16, 8))
    fig.subplots_adjust(wspace=.01, hspace=.02)
    for i, title, data_n, tar in zip([1, 2], titles, DATA, TAR):
        ax = fig.add_subplot(1, 2, i, projection='3d')
        color=['b','r','g','y']
        for j in range(N):
            ax.scatter(data_n[j, 0], data_n[j, 1], data_n[j, 2], c=color[tar[j]], s=35, alpha=0.4, marker='P')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=20., azim=-25)
            ax.set_title(title, fontsize=14, y=0.01)

    plt.show()
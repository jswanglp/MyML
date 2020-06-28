# Метод спуска с учителем (Supervised Descent Method, SDM)
# В программе реализовано приближение с помощью кривых при использовании алгоритма SDM
# Подходящая функция является полиномиальной функцией
# Supervised Descent Method and Its Applications to Face Alignment: https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Xiong_Supervised_Descent_Method_2013_CVPR_paper.pdf
# -*- coding: utf-8 -*- 
"""
Created on 22 June, 2019
@author jswanglp

requirements:
    matplotlib==2.0.2
    numpy==1.15.4
    tensorflow==1.12.0
    scikit_learn==0.23.1

"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

if __name__ == '__main__':

    # Предварительная обработка данных выборки
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
    degree = 8    # наивысший порядок полинома
    num = 30    # количество тестовых данных
    theta = np.mat(np.ones((degree+1, 1)))
    poly = PolynomialFeatures(degree, include_bias=False)
    X_polyfeatures = poly.fit_transform(X)
    X_with_x0 = np.c_[np.ones((X.shape[0],1)), X_polyfeatures]

    # Оптимальные параметры, полученные с помощью метода регуляризации Тихонова
    lamb_ridge = 0.1
    first_half = np.linalg.inv((np.dot(X_with_x0.T,X_with_x0) + lamb_ridge*np.eye(degree+1)))
    sec_half = np.dot(X_with_x0.T, Y)
    theta_pre = np.dot(first_half, sec_half)

    # Настройка параметров
    R = []  # параметры для каждой итерации
    n = 3
    y0 = Y
    theta_start = 0.1 * np.random.randn(degree+1, 1) # начальный параметр

    # Итерация к сходимости
    for j in range(3):
        dx = theta_pre - theta_start
        dy = y0 - np.dot(X_with_x0, theta_start)
        # Метод регуляризации Тихонова
        first_h = np.linalg.inv(np.dot(dy, dy.T) + 1. * np.eye(10))
        sec_h = np.dot(dy, dx.T)
        r_tmp =  np.dot(first_h, sec_h)
        theta_start = theta_start + np.dot(r_tmp.T, dy)
        R.append(r_tmp)
        print(np.linalg.norm(dx))    # ошибки
    print('--------------------')
    x_pre = np.linspace(x.min(), x.max(), num)    # генерование больше данных для прогнозирования в диапазоне, заданном обучающими данными
    X_pre = x_pre.reshape(-1, 1)
    X_pre_polyfeatures = poly.fit_transform(X_pre)
    X_pre_with_x0 = np.c_[np.ones((X_pre.shape[0],1)), X_pre_polyfeatures]
    y_pre = np.dot(X_pre_with_x0, theta_start)    # прогнозирование y_pre

    # Представление аппроксимирующих кривых
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y, '*r', markersize=8, label='Original datas')
    ax.plot(X_pre_with_x0[:,1], y_pre, 'b', label='Predictive function')
    ax.legend(loc='upper right', fontsize=14)
    ax.tick_params(labelsize=14)
    plt.show()
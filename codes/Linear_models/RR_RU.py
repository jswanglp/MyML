# В программе реализована аппроксимация неизвестной функции с помощью метода регуляризации Тихонова
# Количество выборок равно 10, тестовые данные num = 30
# Неизвестная функция - это полиномиальная функция со смещением, представленное theta[0]
# conding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def mse_loss(theta, X, y, lamb): # вычисление среднеквадратических ошибок
	'''
	lamb: коэффициент регуляризации
	theta: (n+1)·1 матрица, x0=1 постоянная величина b
	X: m·(n+1) матрица, содержащая x0
	'''
	y_f = np.dot(X, theta)
	theta_w = theta[1:]
	loss = 0.5 * mean_squared_error(y_f, y) + 0.5 * lamb * np.sum(np.square(theta_w))
	return loss
	
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
	theta = np.ones((degree+1, 1))
	poly = PolynomialFeatures(degree, include_bias=False)
	X_polyfeatures = poly.fit_transform(X)
	X_with_x0 = np.c_[np.ones((X.shape[0],1)), X_polyfeatures]
	
	# Обучение модели
	Loss = []
	lamb_set = [0., .01, 0.1, 1., 20]
	for i, lamb_ridge in enumerate(lamb_set):
		first_half = np.linalg.inv((np.dot(X_with_x0.T,X_with_x0) + lamb_ridge*np.eye(degree+1)))
		sec_half = np.dot(X_with_x0.T, Y)
		theta_pre = np.dot(first_half, sec_half)
		if i == 0: theta0 = theta_pre
		y_train = np.dot(X_with_x0, theta_pre)    # прогнозирование
		
		x_pre = np.linspace(x.min(), x.max(), num)    # генерование больше данных для прогнозирования в диапазоне, заданном обучающими данными
		X_pre = x_pre.reshape(-1, 1)
		X_pre_polyfeatures = poly.fit_transform(X_pre)
		X_pre_with_x0 = np.c_[np.ones((X_pre.shape[0],1)), X_pre_polyfeatures]
		y_pre = np.dot(X_pre_with_x0, theta_pre)

		loss = mse_loss(theta_pre, X_with_x0,Y, lamb_ridge)    # mse_loss обучающих данных
		Loss.append(loss)

		# Представление аппроксимирующих кривых при использовании различных коэффициентов lamb
		# print('When lamb is %.2f, mse_loss is: %.3f' %(lamb_ridge,loss))
		fig = plt.figure(i, figsize=(8, 6))
		ax = fig.add_subplot(1, 1, 1)
		ax.plot(x,y,'*r', markersize=8, label='Original datas')
		# ind_y = np.argsort(X_with_x0[:,1], axis=0)
		# ax.plot(np.sort(X_with_x0[:,1]), y_pre[ind_y], 'b', label='Predictive function')
		ax.plot(X_pre_with_x0[:,1], y_pre, 'b', label='Predictive function')
		ax.legend(loc='upper right', fontsize=14)
		ax.set_title(r'$\lambda = {0:.2f}, MSEloss = {1:.2f} $'.format(lamb_ridge,loss), fontsize=14)
		ax.tick_params(labelsize=14)
		# ax.set_ylabel('y', fontsize=14)
		# ax.set_title(r'$\lambda = {0:.2f}$'.format(lamb_ridge)+r'$, mse_{-}loss = {0:.2f}$'.format(loss), fontsize=14)
		plt.show()
		# print('When lamb is %.2f, mse_loss is: %.3f' %(lamb_ridge, loss))
	print('\n','Loss list: ', Loss)
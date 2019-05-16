# 混合高斯模型类
import numpy as np

class GaussianMM:
    
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.alpha = None
        self.f_dim = None
        self.num_mixed = None

    # 初始化
    def init_fn(self, f_dim=3, num_mixed=4):
        self.f_dim = f_dim
        self.num_mixed = num_mixed
        self.mu = np.random.randn(num_mixed, f_dim) + 10
        self.sigma = np.zeros((num_mixed, f_dim, f_dim))
        for i in range(num_mixed):
            self.sigma[i, :, :] = np.diag(np.random.randint(10, 25, size=(3, )))
        self.alpha = [1. / num_mixed] * int(num_mixed)
        return 'Initialization completed !'

    # e-step
    def e_step(self, X):
        N, _ = X.shape
        expec = np.zeros((N, self.num_mixed))
        for i in range(N):  
            denom = 0  
            # numer = 0
            F_list = []
            S_list = []
            for j in range(self.num_mixed):
                sig_inv = np.linalg.inv(self.sigma[j, :, :])
                expo_1 = np.matmul(-(X[i, :] - self.mu[j, :]), sig_inv)
                expo_2 = np.matmul(expo_1, ((X[i, :] - self.mu[j, :])).reshape(-1, 1))
                first_half = self.alpha[j] * np.exp(expo_2)
                # first_half = alpha_[j] * np.exp(-(X[i, :] - mu[j, :]) * sig_inv * ((X[i, :] - mu[j, :])).reshape(-1, 1))
                sec_half = np.sqrt(np.linalg.det(np.mat(self.sigma[j, :, :])))
                F_list.append(first_half[0])
                S_list.append(sec_half)
                denom += first_half[0] / sec_half      #分母
            for j in range(self.num_mixed):  
                numer = F_list[j] / S_list[j]        #分子
                expec[i, j]= numer / denom      #求期望
        return expec

    # m-step
    def m_step(self, X, expec):  
        N, c = X.shape
        lemda = 1e-15
        for j in range(self.num_mixed):  
            denom = 0   #分母  
            numer = 0   #分子 
            sig = 0 
            for i in range(N):  
                numer += expec[i, j] * X[i, :]  
                denom += expec[i, j]
            self.mu[j, :] = numer / denom    #求均值  
            for i in range(N):
                x_tran = (X[i, :] - self.mu[j, :]).reshape(-1, 1)
                x_nor = (X[i, :] - self.mu[j, :]).reshape(1, -1)
                sig += expec[i, j] * np.matmul(x_tran, x_nor)
            self.alpha[j] = denom / N        #求混合项系数
            self.sigma[j, :, :] = sig / denom + np.diag(np.array([lemda] * c))
        return self.mu, self.sigma, self.alpha

    # 训练
    def fit(self, X, err_mu=5, err_alpha=0.01, max_iter=100):
        iter_num = 0
        while True:
            if iter_num == max_iter: break
            iter_num += 1
            mu_prev = self.mu.copy()
            # print(mu_prev)
            alpha_prev = self.alpha.copy()
            # print(alpha_prev)
            expec = self.e_step(X)
            self.mu, self.sigma, self.alpha = self.m_step(X, expec)
            print(u"迭代次数:", iter_num)
            print(u"估计的均值:\n", self.mu)
            print(u"估计的混合项系数:", self.alpha, '\n')
            err = abs(mu_prev - self.mu).sum()      #计算误差
            err_a = abs(np.array(alpha_prev) - np.array(self.alpha)).sum()
            if (err < err_mu) and (err_a < err_alpha):     #达到精度退出迭代
                print(u"\n最终误差:", [err, err_a])  
                break
        print('训练已完成 !')
    
    # 预测属于第几个高斯成分
    def predict(self, X):
        expec = self.e_step(X)
        return np.argmax(expec, axis=1)

# 该程序实现 EM 算法对伯努利分布的参数估计
# 数据为 1 维，多维数据请参考混合高斯模型
# 参考程序：https://github.com/wzyonggege/statistical-learning-method/blob/master/EM/em.ipynb
# -*- coding: utf-8 -*- 
"""
Created on 13 May, 2019
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

# EM 类
class EM:
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob
        
    # e_step
    def pro_1(self, x):
        return self.pro_A * np.power(self.pro_B, x) * np.power((1 - self.pro_B), 1 - x)
    def pro_2(self, x):
        return (1 - self.pro_A) * np.power(self.pro_C, x) * np.power((1 - self.pro_C), 1 - x)
    def pmf(self, x):
        return self.pro_1(x) / (self.pro_1(x) + self.pro_2(x))
    
    # m_step
    def fit(self, data, max_error=1e-5):
        count = data.shape[0]
        print('init prob:{}, {}, {}'.format(self.pro_A, self.pro_B, self.pro_C))
        d = 0
        while True:
            d += 1
            PMF = [self.pmf(x) for x in data]
            pro_A = 1/ count * sum(PMF)
            pro_B = sum([PMF[k] * data[k] for k in range(count)]) / sum([PMF[k] for k in range(count)])
            pro_C = sum([(1 - PMF[k]) * data[k] for k in range(count)]) / sum([(1 - PMF[k]) for k in range(count)])
            error = abs(pro_A - self.pro_A) + abs(pro_B - self.pro_B) + abs(pro_C - self.pro_C)
            print_list = [d, pro_A, pro_B, pro_C, error]
            print('Step {0[0]},  pro_a:{0[1]:.3f}, pro_b:{0[2]:.3f}, pro_c:{0[3]:.3f}, error: {0[4]:.6f}.'.format(print_list))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C
            if error < max_error: break
    def mlf(self, y):
        return self.pro_1(y) + self.pro_2(y)
           
# 生成一维伯努利分布的数据
data = np.random.binomial(1, 0.2, size=[200, ])
em = EM(prob=[0.2, 0.3, 0.4])
em.fit(data)

# >> init prob:0.2, 0.3, 0.4
# >> Step 1,  pro_a:0.212, pro_b:0.149, pro_c:0.214, error: 0.349631.
# >> Step 2,  pro_a:0.212, pro_b:0.149, pro_c:0.214, error: 0.000000.

em.mlf(0)
# >> 0.8000000000000007

em.mlf(1)
# >> 0.19999999999999934
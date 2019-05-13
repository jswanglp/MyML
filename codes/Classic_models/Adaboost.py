# Adaboost 分类器类
import numpy as np

class AdaBoost:
    def __init__(self, n_estimators=10, learning_rate=1.0):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate
    
    def init_args(self, datasets, labels):
        
        self.X = datasets
        self.Y = labels.flatten()
        self.M, self.N = datasets.shape
        
        # 弱分类器数目和集合
        self.clf_sets = []
        
        # 初始化 weights
        self.weights = [1.0 / self.M] * self.M
        
        # G(x)系数 alpha
        self.alpha = []
    
    # G(x) 为基础分类器加权和，基础分类器 -- y = direct * sign(x - v) 
    def G_fn(self, features, labels, weights):
        m = len(features)
        error = 100000.0 # 无穷大
        best_v = 0.0
        # 计算 features 每个维度特征的分类误差
        features_min = min(features)
        features_max = max(features)
        n_step = (features_max - features_min + self.learning_rate) // self.learning_rate
        # print('n_step:{}'.format(n_step))
        direct, compare_array = None, None
        for i in range(1, int(n_step)):
            v = features_min + self.learning_rate * i
            
            if v not in features:
                # 误分类计算
                compare_array_positive = np.array([1 if features[k] > v else -1 for k in range(m)])
                weight_error_positive = sum([weights[k] for k in range(m) if compare_array_positive[k] != labels[k]])
                
                compare_array_nagetive = np.array([-1 if features[k] > v else 1 for k in range(m)])
                weight_error_nagetive = sum([weights[k] for k in range(m) if compare_array_nagetive[k] != labels[k]])

                if weight_error_positive < weight_error_nagetive:
                    weight_error = weight_error_positive
                    _compare_array = compare_array_positive
                    direct = 'positive'
                else:
                    weight_error = weight_error_nagetive
                    _compare_array = compare_array_nagetive
                    direct = 'nagetive'
                    
                # print('v:{} error:{}'.format(v, weight_error))
                if weight_error < error:
                    error = weight_error
                    compare_array = _compare_array
                    best_v = v
        return best_v, direct, error, compare_array
        
    # 计算 alpha
    def alpha_fn(self, error):
        return 0.5 * np.log((1 - error) / error)
    
    # 规范化因子
    def Z_fn(self, weights, a, clf):
        return sum([weights[i]*np.exp(-a * self.Y[i] * clf[i]) for i in range(self.M)])
        
    # 权值更新
    def w_fn(self, a, clf, Z):
        for i in range(self.M):
            self.weights[i] = self.weights[i] * np.exp(-a * self.Y[i] * clf[i]) / Z
    
    # G(x)的线性组合
    def f_fn(self, alpha, clf_sets):
        pass
    
    # 基础分类器, v 为阈值
    def G(self, x, v, direct):
        if direct == 'positive':
            return 1 if x > v else -1 
        else:
            return -1 if x > v else 1 
    
    def fit(self, X, y):
        self.init_args(X, y)
        
        for epoch in range(self.clf_num):
            best_clf_error, best_v, clf_result = 100000, None, None
            # 根据特征维度, 选择误差最小的
            for j in range(self.N):
                features = self.X[:, j]
                # 分类阈值，分类误差，分类结果
                v, direct, error, compare_array = self.G_fn(features, self.Y, self.weights)
                
                if error < best_clf_error:
                    best_clf_error = error
                    best_v = v
                    final_direct = direct
                    clf_result = compare_array
                    axis = j
                print_list = [epoch + 1, self.clf_num, j, error, best_v]
                print('epoch:{0[0]}/{0[1]}, feature:{0[2]}, error:{0[3]:.3f}, v:{0[4]:.3f}'.format(print_list))
                if best_clf_error == 0:
                    best_clf_error = 1e-10
                    break
                
            # 计算 G(x) 系数 alpha
            a = self.alpha_fn(best_clf_error)
            self.alpha.append(a)
            # 记录分类器
            self.clf_sets.append((axis, best_v, final_direct))
            # 规范化因子
            Z = self.Z_fn(self.weights, a, clf_result)
            # 权值更新
            self.w_fn(a, clf_result, Z)
            print_list = [epoch + 1, self.clf_num, best_clf_error, best_v, final_direct, a]
            print('classifier:{0[0]}/{0[1]}, error:{0[2]:.3f}, v:{0[3]:.3f}, direct:{0[4]}, a:{0[5]:.3f}'.format(print_list))
            # print('weight:{}'.format(self.weights))
            print('\n')

    def corr_fn(self):
        L = [tup[-1] for tup in self.clf_sets]
        num_p = L.count('positive')
        return 1 if num_p < 2 else -1

    def predict(self, feature):
        result = 0.0
        for i in range(len(self.clf_sets)):
            axis, clf_v, direct = self.clf_sets[i]
            f_input = feature[axis]
            result += self.alpha[i] * self.G(f_input, clf_v, direct)
        # sign
        return np.sign(result) * self.corr_fn()
    
    def score(self, X_test, y_test):
        right_count = 0
        y_test = y_test.flatten()
        for i in range(len(X_test)):
            feature = X_test[i]
            if self.predict(feature) == y_test[i]:
                right_count += 1
        
        return right_count / len(X_test)
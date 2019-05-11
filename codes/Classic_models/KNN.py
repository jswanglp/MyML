# KNN 类
import numpy as np

class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 临近点个数, 最好选奇数
        parameter: p 距离度量
        """
        if n_neighbors % 2 == 0:
            print('n_neighbors 最好为奇数！')
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train.flatten()
    
    def predict(self, X):
        # 取出n个点
        knn_list = []
        for i in range(self.n):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            knn_list.append((dist, self.y_train[i]))
            
        # 遍历得到距离最近的 n 个点    
        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
                
        # 预测类别
        knn = np.array([k[-1] for k in knn_list])
        return np.sign(knn.sum()) if knn.sum() != 0 else 1
    
    def score(self, X_test, y_test):
        y_test = y_test.flatten()
        right_count = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / X_test.shape[0]


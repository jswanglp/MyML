# 贝叶斯分类器类
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.model = None
        self.prior = None

    # 数学期望
    @staticmethod
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差（方差）
    def stdev(self, X):
        avg = self.mean(X)
        return np.sqrt(sum([pow(x-avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        exponent = np.exp(-(np.power(x-mean,2)/(2*np.power(stdev,2))))
        return (1 / (np.sqrt(2*np.pi) * stdev)) * exponent

    # 处理X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries
    
    # 计算每类的先验概率
    def prob_y(self, y):
        labels = list(set(y))
        num_labels = np.array([sum((y == label) << 0) for label in labels])
        return num_labels.astype('float32') / y.shape[0]
    
    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label:[] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        self.prior = self.prob_y(y)
        return 'model is trained!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        probabilities = {}
        prob = 0
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(input_data[i], mean, stdev)
            probabilities[label] *= self.prior[int(label)]
            prob += probabilities[label]
        return probabilities, prob
    
    # 归一化概率  
    def normalized_prob(self, input_data):
        probabilities, prob = self.calculate_probabilities(input_data)
        for label, value in probabilities.items():
            probabilities[label] = value / prob
        return probabilities

    # 类别
    def predict(self, X_test):
        label = sorted(self.calculate_probabilities(X_test)[0].items(), key=lambda x: x[-1])[-1][0]
        return label.astype("uint8")
    
    # 计算精度
    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1
        return right / float(len(X_test))
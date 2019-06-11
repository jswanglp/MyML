# MyML
> 温故而知新，可以为师矣<a href='#fn1' name='fn1b'><sup>[1]</sup></a>。
- [русско-английский](README_RU.md)

花了一个多月把之前写的一些关于机器学习、深度学习的模型都重新编译并且跑了一遍，渣渣电脑，幸好有Google的羊毛可以薅，省出很多时间可以把代码的注释翻译成[俄语版本](README_RU.md)，总算快完工了🤣，可以投入新的战场了😊。

## 机器学习 (Machine learning)

#### 线性模型 (Linear models)
- ##### 线性回归 (Linear regression)
    + ##### 梯度下降 (Gradient descent) ([code](codes/Linear_models/linear_regression.py)) ([notebook-in-colab](notebooks(colab)/Linear_models/linear_regression.ipynb))
    + ##### 岭回归 (Ridge regression) ([code](codes/Linear_models/RR.py)) ([notebook-in-colab](notebooks(colab)/Linear_models/RR.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/ridge-regression))
- ##### 逻辑回归 (Logistic regression) ([code](codes/Linear_models/logistic_regression.py)) ([notebook-in-colab](notebooks(colab)/Linear_models/logistic_regression.ipynb))

#### 几种经典模型 (Classic models)
- ##### 主成分分析 (Principal Component Analysis, PCA) ([code](codes/Classic_models/PCA.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/PCA.ipynb))
- ##### K-近邻算法 (k-nearest neighbors algorithm, k-NN) ([code](codes/Classic_models/KNN_main.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/KNN.ipynb))
- ##### 决策树 (Decision tree) ([code](codes/Classic_models/Decision_tree.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/Decision_tree.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/decision-tree))
- ##### 支持向量机模型 (Support Vector Machines, SVM)
    + ##### NN 降维 (Dimensionality reduction with NN) ([code](codes/Classic_models/linear_SVM.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/linear_SVM.ipynb))
    + ##### CNN 降维 (Dimensionality reduction with CNN) ([code](codes/Classic_models/linear_SVM(CNN).py)) ([notebook-in-colab](notebooks(colab)/Classic_models/linear_SVM(CNN).ipynb))
- ##### 概率图模型 (Probabilistic Graphical Model, PGM)
    + ##### 朴素贝叶斯分类器 (Naive Bayes classifier) ([code](codes/Classic_models/NB.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/NB.ipynb))
    + ##### 隐马尔科夫模型 (Hidden Markov Model, HMM) ([code](codes/Classic_models/HMM.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/HMM.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/hmm-gaussian))
    + ##### 最大期望算法 (Expectation Maximization algorithm, EM) ([code](codes/Classic_models/EM.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/EM.ipynb))
    + ##### 混合高斯模型 (Gaussian mixture model, GMM) ([code](codes/Classic_models/GMM.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/GMM.ipynb))
- ##### 聚类 (Clustering)
    + ##### k-means 聚类 (k-means clustering, k-means) ([code](codes/Classic_models/kmeans.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/kmeans.ipynb))
- ##### 集成学习 (Ensemble learning)
    + ##### 随机森林 (Random Forest, RF) ([code](codes/Classic_models/RF.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/RF.ipynb))
    + ##### AdaBoost 算法 (Adaptive Boosting, AdaBoost) ([code](codes/Classic_models/Adaboost_main.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/Adaboost.ipynb))

#### 神经网络模型 (Neural network models)
- ##### 监督学习模型 (Supervised learning models)
    + ##### 曲线拟合 (Curve fitting) ([code](codes/Neural_network_models/Supervised_learning_models/curve_fitting.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/curve_fitting.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/curve-fitting))
    + ##### 多层感知机 (Multilayer Perceptron, MLP) ([code](codes/Neural_network_models/Supervised_learning_models/MLP.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/MLP.ipynb))
    + ##### 卷积神经网络 (Convolutional Neural Network, CNN) ([code-keras](codes/Neural_network_models/Supervised_learning_models/CNN_keras.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/CNN_keras.ipynb))
    + ##### 卷积神经网络 (Convolutional Neural Network, CNN) ([code-tf](codes/Neural_network_models/Supervised_learning_models/CNN_tf.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/CNN_tf.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/cnn-tf))
    + ##### 基于 CNN 的人脸识别 (Facial recognition based on CNN) ([code](codes/Neural_network_models/Supervised_learning_models/Facial_recognition.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/Facial_recognition.ipynb))
    + ##### 正则化 (Regularization) ([code](codes/Neural_network_models/Supervised_learning_models/Facial_recognition_l2.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/Facial_recognition_l2.ipynb))
    + ##### 批量归一化 (Batch Normalization, BN) ([code](codes/Neural_network_models/Supervised_learning_models/Facial_recognition_bn.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/Facial_recognition_bn.ipynb))
    + ##### 小波神经网络 (Wavelet Neural Network, WNN) ([code](codes/Neural_network_models/Supervised_learning_models/WNN.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/WNN.ipynb))
- ##### 非监督学习模型 (Unsupervised learning models)
    + ##### 自编码器 (Autoencoder, AE) ([code](codes/Neural_network_models/Unsupervised_learning_models/AE.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/AE.ipynb))
    + ##### 变分自编码器 (Variational autoencoder, VAE) ([code](codes/Neural_network_models/Unsupervised_learning_models/VAE.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/VAE.ipynb))
    + ##### 生成对抗网络 (Generative Adversarial Networks, GAN) ([code](codes/Neural_network_models/Unsupervised_learning_models/GAN.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/GAN.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/gan-tf))
    + ##### 深度卷积生成对抗网络 (mnist) (Deep Convolutional Generative Adversarial Networks, DCGAN) ([code](codes/Neural_network_models/Unsupervised_learning_models/DCGAN.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/DCGAN.ipynb))
    + ##### 深度卷积生成对抗网络 (FaceWarehouse) (Deep Convolutional Generative Adversarial Networks, DCGAN) ([code](codes/Neural_network_models/Unsupervised_learning_models/DCGAN_for_faces.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/DCGAN_for_faces.ipynb))

#### 其它 (Others)
- ##### Tensorboard 的使用 (Tensorboard tutorials) ([code](codes/Others/tensorboard_tutorials.py)) ([notebook-in-colab](notebooks(colab)/Others/tensorboard_tutorials.ipynb))

- ##### TensorFlow Dataset 类的使用 (TensorFlow Dataset class tutorials) ([code](codes/Others/Dataset_tutorials.py)) ([notebook-in-colab](notebooks(colab)/Others/Dataset_tutorials.ipynb))

#### 有用的教程 (Useful tutorials)
- ##### [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples) 适用于初学者的 TensorFlow 教程
- ##### [TensorFlow-Course](https://github.com/machinelearningmindset/TensorFlow-Course) 另一个简易教程，包含 `code` 与 ` jupyter notebook`
- ##### [Statistical-learning-method](https://github.com/wzyonggege/statistical-learning-method) 通过 `Python` 实现的一些统计学模型
- ##### [TensorFlow Datasets](https://github.com/tensorflow/datasets) TensorFlow 中数据集类的使用
- ##### [Tensorboard](https://github.com/tensorflow/tensorboard) TensorFlow 可视化工具 Tensorboard 的使用

-----
**脚注 (Footnote)**

<a name='fn1'>[1]</a>： [论语·第二章·为政篇 -- 孔丘](http://www.guoxue.com/book/lunyu/0002.htm)

<a href='#fn1b'><small>↑Back to Content↑</small></a>
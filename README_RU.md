# MyML
> Повторение - мать учения<a href='#fn1' name='fn1b'><sup>[1]</sup></a>.
- [中英文](README.md)

## Машинное обучение (Machine learning)

#### Линейные модели (Linear models)
- ##### Линейная регрессия (Linear regression)
    + ##### Градиентный спуск (Gradient descent) ([code](codes/Linear_models/linear_regression_RU.py)) ([notebook-in-colab](notebooks(colab)/Linear_models/linear_regression_RU.ipynb))
    + ##### Метод регуляризации Тихонова (Ridge regression) ([code](codes/Linear_models/RR_RU.py)) ([notebook-in-colab](notebooks(colab)/Linear_models/RR_RU.ipynb))
- ##### Логистическая регрессия (Logistic regression) ([code](codes/Linear_models/logistic_regression_RU.py)) ([notebook-in-colab](notebooks(colab)/Linear_models/logistic_regression_RU.ipynb))

#### Классические модели (Classic models)
- ##### Метод главных компонент (Principal Component Analysis, PCA) ([code](codes/Classic_models/PCA_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/PCA_RU.ipynb))
- ##### Метод k-ближайших соседей (k-nearest neighbors algorithm, k-NN) ([code](codes/Classic_models/KNN_main_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/KNN_RU.ipynb))
- ##### Дерево принятия решений (Decision tree) ([code](codes/Classic_models/Decision_tree_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/Decision_tree_RU.ipynb))
- ##### Метод опорных векторов (Support Vector Machines, SVM)
    + ##### Снижение с помощью ИНС (Dimensionality reduction with NN) ([code](codes/Classic_models/linear_SVM_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/linear_SVM_RU.ipynb))
    + ##### Снижение с помощью СНС (Dimensionality reduction with CNN) ([code](codes/Classic_models/linear_SVM(CNN).py)) ([notebook-in-colab](notebooks(colab)/Classic_models/linear_SVM(CNN).ipynb))
- ##### Графовая вероятностная модель (Probabilistic Graphical Model, PGM)
    + ##### Наивный байесовский классификатор (Naive Bayes classifier) ([code](codes/Classic_models/NB.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/NB.ipynb))
    + ##### Скрытая марковская модель (Hidden Markov Model, HMM)
    + ##### EM-алгоритм (Expectation Maximization algorithm, EM) ([code](codes/Classic_models/EM.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/EM.ipynb))
    + ##### Модель гауссовых смесей (Gaussian mixture model, GMM) ([code](codes/Classic_models/GMM.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/GMM.ipynb))
- ##### Кластеризация (Clustering)
    + ##### Метод k-средних (k-means clustering, k-means) ([code](codes/Classic_models/kmeans.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/kmeans.ipynb))
- ##### Ансамблевое обучение (Ensemble learning)
    + ##### Случайный лес (Random Forest, RF) ([code](codes/Classic_models/RF.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/RF.ipynb))
    + ##### Алгоритм AdaBoost (Adaptive Boosting, AdaBoost) ([code](codes/Classic_models/Adaboost_main.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/Adaboost.ipynb))

#### Модели нейронных сетей (Neural network models)
- ##### Модели обучения с учителем (Supervised learning models)
    + ##### Многослойный персептрон (Multilayer Perceptron, MLP) ([code](codes/Neural_network_models/Supervised_learning_models/MLP.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/MLP.ipynb))
    + ##### Сверточные нейронные сети (Convolutional Neural Network, CNN) ([code-keras](codes/Neural_network_models/Supervised_learning_models/CNN_keras.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/CNN_keras.ipynb))
    + ##### Сверточные нейронные сети (Convolutional Neural Network, CNN) ([code-tf](codes/Neural_network_models/Supervised_learning_models/CNN_tf.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/CNN_tf.ipynb))
    + ##### Распознавание лиц на основе СНС (Facial recognition based on CNN) ([code](codes/Neural_network_models/Supervised_learning_models/Facial_recognition.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/Facial_recognition.ipynb))
    + ##### Регуляризация (Regularization) ([code](codes/Neural_network_models/Supervised_learning_models/Facial_recognition_l2.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/Facial_recognition_l2.ipynb))
    + ##### Батч-нормализация (Batch Normalization, BN) ([code](codes/Neural_network_models/Supervised_learning_models/Facial_recognition_bn.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/Facial_recognition_bn.ipynb))
- ##### Модели обучения без учителя (Unsupervised learning models)
    + ##### Автокодировщик (Autoencoder, AE) ([code](codes/Neural_network_models/Unsupervised_learning_models/AE.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/AE.ipynb))
    + ##### Вариационный автокодировщик (Variational autoencoder, VAE) ([code](codes/Neural_network_models/Unsupervised_learning_models/VAE.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/VAE.ipynb))
    + ##### Генеративно-состязательные сети (Generative Adversarial Networks, GAN) ([code](codes/Neural_network_models/Unsupervised_learning_models/GAN.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/GAN.ipynb))
    + ##### Глубокие сверточные генеративно-состязательные сети (Deep Convolutional Generative Adversarial Networks, DCGAN)

#### Другие (Others)
- ##### Пособия tensorboard (Tensorboard tutorials) ([code](codes/Others/tensorboard_tutorials.py)) ([notebook-in-colab](notebooks(colab)/Others/tensorboard_tutorials.ipynb))
- ##### Пособия TensorFlow Dataset class (TensorFlow Dataset class tutorials) ([code](codes/Others/Dataset_tutorials.py)) ([notebook-in-colab](notebooks(colab)/Others/Dataset_tutorials.ipynb))

-----
**Сноска (Footnote)**

<a name='fn1'>[1]</a>： [Пословицы русского народа -- В.И. Даля](http://dslov.ru/txt/81/t81_168.htm)

<a href='#fn1b'><small>↑Вернуться к содержанию↑</small></a>
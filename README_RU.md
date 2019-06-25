# MyML
> Повторение - мать учения<a href='#fn1' name='fn1b'><sup>[1]</sup></a>.
- [中英文](README.md)

Мне потребовалось больше месяца, чтобы перекомпилировать и запустить некоторые из кодов моделей, которые я написал о машинном обучении и глубоком обучении. Видеокарта моего компьютера слишком плохая, к счастью, Google предоставляет нам GPU бесплатно. Это экономит много времени, и я могу перевести комментарии кодов на [русский язык](README_RU.md). Этот репозиторий почти закончен🤣, и пришло время готовиться к новому полю боя😊.  
Некоторые другие полезные репозитории:  

## Машинное обучение (Machine learning)

#### Линейные модели (Linear models)

- ##### Линейная регрессия (Linear regression)
    + ##### Градиентный спуск (Gradient descent) ([code](codes/Linear_models/linear_regression_RU.py)) ([notebook-in-colab](notebooks(colab)/Linear_models/linear_regression_RU.ipynb)) Другой пример сравнения между [алгоритмом градиентного спуска](https://en.wikipedia.org/wiki/Gradient_descent) и [алгоритмом Левенберга-Марквардта](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) приведен в этом [repo](https://github.com/jswanglp/Levenberg-Marquardt-algorithm/blob/master/README.pdf).
    + ##### Метод регуляризации Тихонова (Ridge regression) ([code](codes/Linear_models/RR_RU.py)) ([notebook-in-colab](notebooks(colab)/Linear_models/RR_RU.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/ridge-regression))
    + ##### Метод спуска с учителем (Supervised Descent Method, SDM) ([code](codes/Linear_models/SDM_RU.py)) ([notebook-in-colab](notebooks(colab)/Linear_models/SDM_RU.ipynb)) Хотя это не относится к линейной модели, но оптимальные параметры в этом примере найдены методом регуляризации Тихонова.
- ##### Логистическая регрессия (Logistic regression) ([code](codes/Linear_models/logistic_regression_RU.py)) ([notebook-in-colab](notebooks(colab)/Linear_models/logistic_regression_RU.ipynb))

#### Классические модели (Classic models)

- ##### Метод главных компонент (Principal Component Analysis, PCA) ([code](codes/Classic_models/PCA_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/PCA_RU.ipynb))
- ##### Метод k-ближайших соседей (k-nearest neighbors algorithm, k-NN) ([code](codes/Classic_models/KNN_main_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/KNN_RU.ipynb))
- ##### Дерево принятия решений (Decision tree) ([code](codes/Classic_models/Decision_tree_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/Decision_tree_RU.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/decision-tree))
- ##### Метод опорных векторов (Support Vector Machines, SVM)
    + ##### Снижение с помощью ИНС (Dimensionality reduction with NN) ([code](codes/Classic_models/linear_SVM_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/linear_SVM_RU.ipynb))
    + ##### Снижение с помощью СНС (Dimensionality reduction with CNN) ([code](codes/Classic_models/linear_SVM(CNN)_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/linear_SVM(CNN)_RU.ipynb))
- ##### Графовая вероятностная модель (Probabilistic Graphical Model, PGM)
    + ##### Наивный байесовский классификатор (Naive Bayes classifier) ([code](codes/Classic_models/NB_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/NB_RU.ipynb))
    + ##### Скрытая марковская модель (Hidden Markov Model, HMM) (Hidden Markov Model, HMM) ([code](codes/Classic_models/HMM_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/HMM_RU.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/hmm-gaussian))
    + ##### EM-алгоритм (Expectation Maximization algorithm, EM) ([code](codes/Classic_models/EM_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/EM_RU.ipynb))
    + ##### Модель гауссовых смесей (Gaussian mixture model, GMM) ([code](codes/Classic_models/GMM_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/GMM_RU.ipynb))
- ##### Кластеризация (Clustering)
    + ##### Метод k-средних (k-means clustering, k-means) ([code](codes/Classic_models/kmeans_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/kmeans_RU.ipynb))
- ##### Ансамблевое обучение (Ensemble learning)
    + ##### Случайный лес (Random Forest, RF) ([code](codes/Classic_models/RF_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/RF_RU.ipynb))
    + ##### Алгоритм AdaBoost (Adaptive Boosting, AdaBoost) ([code](codes/Classic_models/Adaboost_main_RU.py)) ([notebook-in-colab](notebooks(colab)/Classic_models/Adaboost_RU.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/adaboost))

#### Модели нейронных сетей (Neural network models)

- ##### Модели обучения с учителем (Supervised learning models)
    + ##### Приближение с помощью кривых (Curve fitting) ([code](codes/Neural_network_models/Supervised_learning_models/curve_fitting_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/curve_fitting_RU.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/curve-fitting))
    + ##### Многослойный персептрон (Multilayer Perceptron, MLP) ([code](codes/Neural_network_models/Supervised_learning_models/MLP_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/MLP_RU.ipynb))
    + ##### Сверточные нейронные сети (Convolutional Neural Network, CNN) ([code-keras](codes/Neural_network_models/Supervised_learning_models/CNN_keras_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/CNN_keras_RU.ipynb))
    + ##### Сверточные нейронные сети (Convolutional Neural Network, CNN) ([code-tf](codes/Neural_network_models/Supervised_learning_models/CNN_tf_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/CNN_tf_RU.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/cnn-tf))
    + ##### Распознавание лиц на основе СНС (Facial recognition based on CNN) ([code](codes/Neural_network_models/Supervised_learning_models/Facial_recognition_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/Facial_recognition_RU.ipynb))
    + ##### Регуляризация (Regularization) ([code](codes/Neural_network_models/Supervised_learning_models/Facial_recognition_l2_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/Facial_recognition_l2_RU.ipynb))
    + ##### Батч-нормализация (Batch Normalization, BN) ([code](codes/Neural_network_models/Supervised_learning_models/Facial_recognition_bn_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/Facial_recognition_bn_RU.ipynb))
    + ##### Вейвлет нейронные сети (Wavelet Neural Network, WNN) ([code](codes/Neural_network_models/Supervised_learning_models/WNN_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Supervised_learning_models/WNN_RU.ipynb))
- ##### Модели обучения без учителя (Unsupervised learning models)
    + ##### Автокодировщик (Autoencoder, AE) ([code](codes/Neural_network_models/Unsupervised_learning_models/AE_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/AE_RU.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/autoencoder))
    + ##### Вариационный автокодировщик (Variational autoencoder, VAE) ([code](codes/Neural_network_models/Unsupervised_learning_models/VAE_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/VAE_RU.ipynb))
    + ##### Генеративно-состязательные сети (Generative Adversarial Networks, GAN) ([code](codes/Neural_network_models/Unsupervised_learning_models/GAN_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/GAN_RU.ipynb)) ([kaggle-kernel](https://www.kaggle.com/jswanglp/gan-tf))
    + ##### Глубокие сверточные генеративно-состязательные сети (mnist) (Deep Convolutional Generative Adversarial Networks, DCGAN) ([code](codes/Neural_network_models/Unsupervised_learning_models/DCGAN_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/DCGAN_RU.ipynb))
    + ##### Глубокие сверточные генеративно-состязательные сети (FaceWarehouse) (Deep Convolutional Generative Adversarial Networks, DCGAN) ([code](codes/Neural_network_models/Unsupervised_learning_models/DCGAN_for_faces_RU.py)) ([notebook-in-colab](notebooks(colab)/Neural_network_models/Unsupervised_learning_models/DCGAN_for_faces_RU.ipynb))

#### Другие (Others)

- ##### Учебные пособия tensorboard (Tensorboard tutorials) ([code](codes/Others/tensorboard_tutorials_RU.py)) ([notebook-in-colab](notebooks(colab)/Others/tensorboard_tutorials_RU.ipynb))
- ##### Учебные пособия TensorFlow Dataset class (TensorFlow Dataset class tutorials) ([code](codes/Others/Dataset_tutorials_RU.py)) ([notebook-in-colab](notebooks(colab)/Others/Dataset_tutorials_RU.ipynb))
- ##### Применение VGG сетей в задаче классификации изображений (Application of VGG networks in image classification task) ([code](codes/Others/VGG16_RU.py)) ([notebook-in-colab](notebooks(colab)/Others/VGG16_RU.ipynb))

#### Полезные учебные пособия (Useful tutorials)

- ##### [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples) Учебное пособие по TensorFlow для начинающих
- ##### [TensorFlow-Course](https://github.com/machinelearningmindset/TensorFlow-Course) Другое простое учебное пособие, которое включает в себя `code` и ` jupyter notebook`
- ##### [Statistical-learning-method](https://github.com/wzyonggege/statistical-learning-method) Реализация некоторых статистических моделей при использовании `Python`
- ##### [TensorFlow Datasets](https://github.com/tensorflow/datasets) Учебное пособие TensorFlow Datasets
- ##### [Tensorboard](https://github.com/tensorflow/tensorboard) Использование инструмента визуализации TensorFlow Tensorboard
- ##### [Levenberg-Marquardt-algorithm](https://github.com/jswanglp/Levenberg-Marquardt-algorithm) Сравнение градиентного спуска и алгоритма Левенберга-Марквардта
- ##### [NN_and_WNN](https://github.com/jswanglp/NN_and_WNN) Реализация нейронных сетей и вейвлет нейронных сетей с помощью метода обратного распространения ошибки

-----
**Сноска (Footnote)**

<a name='fn1'>[1]</a>： [Пословицы русского народа -- В.И. Даля](http://dslov.ru/txt/81/t81_168.htm)

<a href='#fn1b'><small>↑Вернуться к содержанию↑</small></a>
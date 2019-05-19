# В программе реализован метод опорных векторов с помощью градиентного спуска
# Оригинальная программа：https://en.wikipedia.org/wiki/Support-vector_machine
# code: utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == '__main__':

    # Настройка параметров
    tf.app.flags.DEFINE_integer('batch_size', 128, 'Number of samples per batch.')
    tf.app.flags.DEFINE_integer('num_epochs', 100, 'Number of epochs for training.')
    tf.app.flags.DEFINE_boolean('is_evaluation', True, 'Whether or not the model should be evaluated.')
    tf.app.flags.DEFINE_float('C_param', 0.1, 'penalty parameter of the error term.')
    tf.app.flags.DEFINE_float('Reg_param', 1.0, 'penalty parameter of the error term.')
    tf.app.flags.DEFINE_float('delta', 1.0, 'The parameter set for margin.')
    tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'The initial learning rate for optimization.')

    FLAGS = tf.app.flags.FLAGS
    
    # Функция ошибки и функция точности
    def loss_fn(W,b,x_data,y_target):
        logits = tf.subtract(tf.matmul(x_data, W), b)
        norm_term = tf.divide(tf.reduce_sum(tf.multiply(tf.transpose(W),W)), 2)
        classification_loss = tf.reduce_mean(tf.maximum(0., tf.subtract(FLAGS.delta, tf.multiply(logits, y_target))))
        total_loss = tf.add(tf.multiply(FLAGS.C_param, classification_loss), tf.multiply(FLAGS.Reg_param, norm_term))
        return total_loss

    def inference_fn(W,b,x_data,y_target):
        prediction = tf.sign(tf.subtract(tf.matmul(x_data, W), b))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
        return accuracy
    
    # Извлечение индексов изображений 0 или 1
    def extraction_fn(data): 
        index_list = []
        for idx in range(data.shape[0]):
            if data[idx] == 0 or data[idx] == 1:
                index_list.append(idx)
        return index_list

    # Обучающие и тестовые изображения
    mnist = input_data.read_data_sets("sample_data/MNIST_data", reshape=True, one_hot=False)
    data = {}
    data['train_image'] = mnist.train.images
    data['train_label'] = mnist.train.labels
    data['test_image'] = mnist.test.images
    data['test_label'] = mnist.test.labels

    index_list_train = extraction_fn(data['train_label'])
    index_list_test = extraction_fn(data['test_label'])

    data['train_image'] = mnist.train.images[index_list_train]
    data['train_label'] = mnist.train.labels[index_list_train]
    data['test_image'] = mnist.test.images[index_list_test]
    data['test_label'] = np.array(mnist.test.labels[index_list_test], dtype=np.float32)
    # data['test_label'] = mnist.test.labels[index_list_test].astype('float32')

    data['train_image_label'] = np.c_[data['train_image'], data['train_label']]
    num_samples, num_features = data['train_image'].shape
    num_hidden = 256

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('Input'):
            x_data = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
            y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # Снижение размерности входных данных
        with tf.name_scope('Net'):
            w_1 = tf.Variable(tf.random_normal(shape=[num_features, num_hidden]), name='w_1')
            b_1 = tf.Variable(tf.random_normal(shape=[num_hidden]), name='b_1')
            layer_1 = tf.sigmoid(tf.matmul(x_data, w_1) + b_1)

            W = tf.Variable(tf.random_normal(shape=[num_hidden, 1]), name='weights')
            b = tf.Variable(tf.random_normal(shape=[1]), name='bias')
        with tf.name_scope('Loss'):
            total_loss = loss_fn(W, b, layer_1, y_target)
        with tf.name_scope('Accuracy'):
            accuracy = inference_fn(W, b, layer_1, y_target)
        with tf.name_scope('Train'):
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)
    
    # Обучение сетей
    test_label = data['test_label'].reshape(-1, 1)
    test_label[test_label==0] = -1
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.num_epochs):
            np.random.shuffle(data['train_image_label'])
            image_batch = data['train_image_label'][:FLAGS.batch_size,:-1]
            label_batch = data['train_image_label'][:FLAGS.batch_size,-1]
            label_batch[label_batch==0] = -1

            _, loss, acc = sess.run([train_op, total_loss, accuracy], feed_dict={x_data: image_batch, 
                                                                                y_target: label_batch.reshape(-1, 1)})
            acc *= 100
            if (epoch + 1) % 5 == 0:
                test_loss, test_acc = sess.run([total_loss, accuracy], feed_dict={x_data: data['test_image'], 
                                                                                y_target: test_label})
                test_acc *= 100
                print_list = [epoch + 1, loss, acc, test_acc]
                print('Epoch {0[0]}, loss: {0[1]:.2f}, training accuracy: {0[2]:.2f}%.'.format(print_list))
                print(' '*10, 'Testing accuracy is {0[3]:.2f}%.'.format(print_list))
    sess.close()







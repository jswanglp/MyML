# В программе реализовано применение глубоких сверточных генеративно-состязательных сетей с тремя сверточными слоями в наборе данных лиц FaceWarehouse
# Размер входных изображений: 45 * 38
# Оригинальная программа: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/dcgan.ipynb
# Литература: https://arxiv.org/pdf/1511.06434.pdf
# -*- coding: utf-8 -*- 
"""
Created on 21 May, 2019
@author jswanglp

requirements:
    scipy==1.1.0
    numpy==1.15.4
    matplotlib==2.0.2
    tensorflow==1.12.0

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io as scio

tf.logging.set_verbosity(tf.logging.ERROR)

# Функция активации LeakyReLU
def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

# Генератор
def generator(x, batch_size, noise_dim, num_neuron, is_training, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse): # установить reuse на True при повторном вызове
        with tf.name_scope('FC'):
            w_fc = tf.get_variable(name='weights_fc', shape=[noise_dim, num_neuron], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_fc = tf.get_variable(name='bias_fc', initializer=tf.constant(0.1, shape=[num_neuron]))
            layer_1 = tf.matmul(x, w_fc) + b_fc
            layer_1_bn = tf.layers.batch_normalization(layer_1, training=is_training) # батч-нормализация1
            layer_1_op = tf.nn.relu(layer_1_bn)
        with tf.name_scope('Conv1'):
            x_imgs = tf.reshape(layer_1_op, shape=[batch_size, 6, 5, 64], name='layer1_imgs')
            w_c1 = tf.get_variable(name='weights_c1', shape=[5, 5, 32, 64], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_c1 = tf.get_variable(name='bias_c1', initializer=tf.constant(0.1, shape=[32]))
            layer_c1 = tf.nn.conv2d_transpose(x_imgs, w_c1, output_shape=[batch_size, 12, 10, 32], strides=[1, 2, 2, 1], padding='SAME') + b_c1
            layer_c1_bn = tf.layers.batch_normalization(layer_c1, training=is_training) # батч-нормализация2
            layer_c1_op = tf.nn.relu(layer_c1_bn)
#             layer_c1_op = tf.nn.relu(layer_c1)
        with tf.name_scope('Conv2'):
            w_c2 = tf.get_variable(name='weights_c2', shape=[5, 5, 16, 32], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_c2 = tf.get_variable(name='bias_c2', initializer=tf.constant(0.1, shape=[16]))
            layer_c2 = tf.nn.conv2d_transpose(layer_c1_op, w_c2, output_shape=[batch_size, 23, 19, 16], strides=[1, 2, 2, 1], padding='SAME') + b_c2
            layer_c2_bn = tf.layers.batch_normalization(layer_c2, training=is_training) # батч-нормализация3
            layer_c2_op = tf.nn.relu(layer_c2_bn)
        with tf.name_scope('Conv3'):
            w_c3 = tf.get_variable(name='weights_c3', shape=[5, 5, 1, 16], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_c3 = tf.get_variable(name='bias_c3', initializer=tf.constant(0.1, shape=[1]))
            layer_c3 = tf.nn.conv2d_transpose(layer_c2_op, w_c3, output_shape=[batch_size, 45, 38, 1], strides=[1, 2, 2, 1], padding='SAME') + b_c3
        with tf.name_scope('Output'):
            x_op = tf.nn.tanh(layer_c3, name='output_gen') # диапазон значений на выходе составляет [-1, 1]
    return x_op

# Дискриминатор
def discriminator(x, is_training, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        with tf.name_scope('Conv1'):
            w_c1 = tf.get_variable(name='weights_c1', shape=[5, 5, 1, 64], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_c1 = tf.get_variable(name='bias_c1', initializer=tf.constant(0.1, shape=[64]))
            layer_c1 = tf.nn.conv2d(x, w_c1, strides=[1, 2, 2, 1], padding='SAME') + b_c1
            layer_c1_bn = tf.layers.batch_normalization(layer_c1, training=is_training) # батч-нормализация1
            layer_c1_op = leakyrelu(layer_c1_bn)
#             layer_c1_op = tf.nn.relu(layer_c1_bn)
        with tf.name_scope('Conv2'):
            w_c2 = tf.get_variable(name='weights_c2', shape=[5, 5, 64, 128], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_c2 = tf.get_variable(name='bias_c2', initializer=tf.constant(0.1, shape=[128]))
            layer_c2 = tf.nn.conv2d(layer_c1_op, w_c2, strides=[1, 2, 2, 1], padding='SAME') + b_c2
            layer_c2_bn = tf.layers.batch_normalization(layer_c2, training=is_training) # батч-нормализация2
            layer_c2_op = leakyrelu(layer_c2_bn)
        with tf.name_scope('Conv3'):
            w_c3 = tf.get_variable(name='weights_c3', shape=[5, 5, 128, 256], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_c3 = tf.get_variable(name='bias_c3', initializer=tf.constant(0.1, shape=[256]))
            layer_c3 = tf.nn.conv2d(layer_c2_op, w_c3, strides=[1, 2, 2, 1], padding='SAME') + b_c3
            layer_c3_bn = tf.layers.batch_normalization(layer_c3, training=is_training) # батч-нормализация3
            layer_c3_op = leakyrelu(layer_c3_bn)
#             layer_c1_op = tf.nn.relu(layer_c2_bn)
            layer_c3_fla = tf.layers.flatten(layer_c3_op)
        with tf.name_scope('FC'):
            num_f = layer_c3_fla.get_shape().as_list()[-1]
            w_fc = tf.get_variable(name='weights_fc', shape=[num_f, 1024], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_fc = tf.get_variable(name='bias_fc', initializer=tf.constant(0.1, shape=[1024]))
            layer_1 = tf.matmul(layer_c3_fla, w_fc) + b_fc
            layer_1_bn = tf.layers.batch_normalization(layer_1, training=is_training) # батч-нормализация4
            # layer_1_op = leakyrelu(layer_1_bn)
            layer_1_op = tf.nn.relu(layer_1_bn)
        # with tf.name_scope('FC2'):
        #     w_fc2 = tf.get_variable(name='weights_fc2', shape=[1024, 512], initializer=tf.initializers.truncated_normal(stddev=0.1))
        #     b_fc2 = tf.get_variable(name='bias_fc2', initializer=tf.constant(0.1, shape=[512]))
        #     layer_2 = tf.matmul(layer_1_op, w_fc2) + b_fc2
        #     layer_2_bn = tf.layers.batch_normalization(layer_2, training=is_training) # батч-нормализация4
        #     # layer_2_op = leakyrelu(layer_2_bn)
        #     layer_2_op = tf.nn.relu(layer_2_bn)
        with tf.name_scope('Output'):
            w_fct = tf.get_variable(name='weights_fct', shape=[1024, 2], initializer=tf.initializers.truncated_normal(stddev=0.1))
            b_fct = tf.get_variable(name='bias_fct', initializer=tf.constant(0.1, shape=[2]))
            layer_2 = tf.matmul(layer_1_op, w_fct) + b_fct
    return layer_2

if __name__ == '__main__':

    # Настройка параметров сетей
    num_epochs = 350 #@param {type: "integer"}
    batch_size = 128 #@param {type: "integer"}
    lr_generator = 8e-4 #@param {type: "number"}
    lr_discriminator = 2e-3 #@param {type: "number"}
    # Network Params
    noise_dim = 100 # размерность шума
    num_neuron = 6 * 5 * 64
    data_path = 'GoogleDrive/My Drive/MATLAB/face recognition/faces_database/faces_150_small.mat'
    data = scio.loadmat(data_path)
    imgs = np.vstack([data['train_faces'], data['test_faces']])
    event_path = './Tensorboard'

    # Структура graph
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('Placeholder'):
            noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='noise_input')
            real_image_input = tf.placeholder(tf.float32, shape=[None, 45, 38, 1], name='image_input')
            batch_s = tf.placeholder(tf.int32)
            is_training = tf.placeholder(tf.bool)

        with tf.name_scope('Network'):
            gen_sample = generator(noise_input, batch_s, noise_dim, num_neuron, is_training)

            disc_real = discriminator(real_image_input, is_training)
            disc_fake = discriminator(gen_sample, is_training, reuse=True)
            stacked_gan = discriminator(gen_sample, is_training, reuse=True)

        with tf.name_scope('Loss'):
            disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=disc_real, labels=tf.ones([batch_s], dtype=tf.int32)))
            disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=disc_fake, labels=tf.zeros([batch_s], dtype=tf.int32)))
            disc_loss = disc_loss_real + disc_loss_fake

            gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=stacked_gan, labels=tf.ones([batch_s], dtype=tf.int32)))

        with tf.name_scope('Optimizer'):
            optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_generator, beta1=0.5, beta2=0.999)
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_discriminator, beta1=0.5, beta2=0.999)

            # Определить переменные, которые необходимо обновлять отдельно для каждого epoch
            # Список переменных генеративных сетей
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Network/Generator')
            # Список переменных дискриминатных сетей
            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Network/Discriminator')

        with tf.name_scope('Train'):
            # TensorFlow UPDATE_OPS collection собирает все операции батч-нормализации и обновляет moving mean/stddev
            gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
            # `control_dependencies` гарантирует, что `gen_update_ops` будет запущен до `minimize` op (backprop)
            with tf.control_dependencies(gen_update_ops):
                train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
            disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
            with tf.control_dependencies(disc_update_ops):
                train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)
        
        # Summaries
        gen_images = (gen_sample + 1.) / 2.
        tf.summary.image('gen_images', gen_images, collections=['train'])
        tf.summary.image('real_images', real_image_input, collections=['train'])
        tf.summary.scalar('gen_loss', gen_loss, collections=['train'])
        tf.summary.scalar('disc_loss', disc_loss, collections=['train'])
        summ = tf.summary.merge_all('train')


    # Обучение модели
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        sum_writer = tf.summary.FileWriter(event_path)
        sum_writer.add_graph(sess.graph)

        for num in range(num_epochs):
            np.random.shuffle(imgs)
            max_batch = imgs.shape[0] // batch_size
            for num_epoch in range(max_batch):
                batch_x = imgs[num_epoch*batch_size:(num_epoch+1)*batch_size].reshape((-1,45,38,1))
                batch_x = batch_x * 2. - 1.

                # Обучение дискриминатора
                z = np.random.uniform(-1., 1., size=[batch_size, noise_dim]) # генерация шумов
                _, dl = sess.run([train_disc, disc_loss], feed_dict={real_image_input: batch_x, noise_input: z, 
                                                                    batch_s: batch_size, is_training: True})

                # Обучение генератора
                z = np.random.uniform(-1., 1., size=[batch_size, noise_dim]) # генерация изображений от шумов
                _, gl = sess.run([train_gen, gen_loss], feed_dict={noise_input: z, 
                                                                    batch_s: batch_size, is_training: True})

            # Summaries
            rs = sess.run(summ, feed_dict={real_image_input: batch_x, noise_input: z, 
                                            batch_s: batch_size, is_training: True})
            sum_writer.add_summary(rs, global_step=num)

            print_list = [num+1, gl, dl]
            if (num + 1) % 20 == 0 or num == 1:
                print('Epoch {0[0]}: Generator Loss: {0[1]:.4f}, Discriminator Loss: {0[2]:.4f}.'.format(print_list))

        # Генерация изображений из шумов путем генеративных сетей
        n = 6
        canvas = np.empty((45 * n, 38 * n))
        for i in range(n):
            z = np.random.uniform(-1., 1., size=[n, noise_dim])
            g = sess.run(gen_sample, feed_dict={noise_input: z, batch_s: n, is_training: False})
            # Отображение сгенерированных изображений на [0,1]
            g_new = (g + 1.) / 2.
            for j in range(n):
                canvas[i * 45:(i + 1) * 45, j * 38:(j + 1) * 38] = g_new[j].reshape([45, 38])

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 12))
        ax.imshow(canvas, cmap='gray')
        ax.set_xticks([]), ax.set_yticks([])
        # img_name1 = os.path.join(event_path, 'generated_images_by_GAN1.jpg')
        # plt.savefig(img_name1)
        plt.show()
    
    sum_writer.close()
    sess.close()
    
    # 36 входные изображений
    n = 6
    canvas = np.empty((45 * n, 38 * n))
    # Отображение сгенерированных изображений на [0,1]
    g_new = (batch_x + 1.) / 2.
    for i in range(n):
        for j in range(n):
            canvas[i * 45:(i + 1) * 45, j * 38:(j + 1) * 38] = g_new[i * 6 + j].reshape([45, 38])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 12))
    ax.imshow(canvas, cmap='gray')
    ax.set_xticks([]), ax.set_yticks([])
    # img_name1 = os.path.join(event_path, 'generated_images_by_GAN1.jpg')
    # plt.savefig(img_name1)
    plt.show()
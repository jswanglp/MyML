# 该程序实现了变分自编码器，给出了隐变量空间在图像空间的投影
# 参考程序：https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/variational_autoencoder.py
# 说明：该程序没有定义 summaries，衰减的 learning_rate，各参数也没有相应的 name_scope。
# -*- coding: utf-8 -*- 
"""
Created on 30 April, 2019
@author jswanglp

requirements:
    scipy==1.1.0
    numpy==1.15.4
    matplotlib==2.0.2
    tensorflow==1.12.0

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mlp

# 载入 mnist 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("sample_data/MNIST", one_hot=True)

# 参数设置
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'learning rate, default is 1e-2.')
tf.app.flags.DEFINE_integer('num_steps', 20000, 'number of epochs, default is 20000.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size, default is 64.')
FLAGS = tf.app.flags.FLAGS

# 网络参数
image_dim = 784
hidden_dim = 512
latent_dim = 2

# Xavier Glorot 参数初始化
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# 定义变量
weights = {
    'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),
    'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([image_dim]))
}

# 构建编码器
input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
encoder = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std'] # z_std=lnσ^2， σ必须为正数

# 生成标准正态分布样本
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
z = z_mean + tf.exp(z_std / 2) * eps # σ = exp(z_std / 2)

# 构建解码器
decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)


# 定义 VAE 的损失函数
def vae_loss(x_reconstructed, x_true):
    # # 重构误差
    # encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
    #                      + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    # encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # mse 误差
    encode_decode_loss = 0.5 * tf.reduce_sum(tf.square(x_reconstructed - x_true))
    # KL 散度误差
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

loss_op = vae_loss(decoder, input_image)
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

# 训练网络
with tf.Session() as sess:

    sess.run(init)

    for i in range(1, FLAGS.num_steps+1):
        batch_x, _ = mnist.train.next_batch(FLAGS.batch_size)

        feed_dict = {input_image: batch_x}
        _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)
        if i % 1000 == 0 or i == 1:
            print('Step %i, Loss: %f' % (i, l))

    # 测试
    # 生成噪声
    noise_input = tf.placeholder(tf.float32, shape=[None, latent_dim])
    # 从噪声解码图像
    decoder = tf.matmul(noise_input, weights['decoder_h1']) + biases['decoder_b1']
    decoder = tf.nn.tanh(decoder)
    decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
    decoder = tf.nn.sigmoid(decoder)

    n = 20
    x_axis = np.linspace(-3, 3, n)
    y_axis = np.linspace(-3, 3, n)
    # 隐变量 z 在图像空间的投影
    canvas = np.empty((28 * n, 28 * n))
    for i, yi in enumerate(x_axis):
        for j, xi in enumerate(y_axis):
            z_mu = np.array([[xi, yi]] * FLAGS.batch_size)
            x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
            canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = \
            x_mean[0].reshape(28, 28)

    mnist_n = input_data.read_data_sets("sample_data/MNIST", one_hot=False)
    imgs, labels = mnist_n.train.images[:1000], mnist_n.train.labels[:1000]
    # 隐变量 z 在隐变量空间的分布
    img_f = sess.run(z, feed_dict={input_image: imgs})
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'cyan', 'yellow', 'gray']
    cmaps = mlp.colors.LinearSegmentedColormap.from_list('mylist',colors, 10)
    fig, AX = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(w=19, h=8)
    fig.subplots_adjust(wspace=0.1)
    ax1, ax2 = AX[0], AX[-1]
    sc = ax2.scatter(img_f[:,0], img_f[:,1], s=32, c=labels, cmap=cmaps, alpha=1.)
    fig.colorbar(sc)
    ax2.grid()
    Xi, Yi = np.meshgrid(x_axis, y_axis)
    ax1.imshow(canvas, origin="upper", cmap="gray")
    ax1.set_xticks(np.linspace(0, 20, 5)*28), ax1.set_yticks(np.linspace(0, 20, 5)*28)
    ax1.set_xticklabels(np.linspace(-3, 3, 5)), ax1.set_yticklabels(np.linspace(3, -3, 5))
    plt.show()

sess.close()
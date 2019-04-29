# 该程序实现了一个简单的线性回归
# coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
# 参数定义
tf.app.flags.DEFINE_float('learning_rate', 3e-2, 'learning rate, default is 3e-2.')
tf.app.flags.DEFINE_integer('num', 32, 'number of samples, default is 32.')
tf.app.flags.DEFINE_integer('num_epochs', 50, 'number of epochs, default is 50.')
FLAGS = tf.app.flags.FLAGS

graph = tf.Graph()
with graph.as_default():

    with tf.name_scope('Input'):
        x_input = tf.placeholder(tf.float32, shape=[None,], name='x_input')
        y_input = tf.placeholder(tf.float32, shape=[None,], name='y_input')
    with tf.name_scope('w_and_b'):
        w = tf.Variable(2.0, name='weight')
        b = tf.Variable(1.0, name='biases')
        y = tf.add(tf.multiply(x_input, w), b)
    with tf.name_scope('Train'):
        loss_op = tf.reduce_mean(tf.pow(y_input - y, 2))
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss_op)

    gradients_node = tf.gradients(loss_op, w)

    # 设置会话
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
 
    # 构造数据集
        x_pure = np.random.randint(-10, 100, FLAGS.num)
        x_train = x_pure + np.random.randn(FLAGS.num) / 10  # 为x加噪声
        y_train = 3 * x_pure + 2 + np.random.randn(FLAGS.num) / 10  # 为y加噪声
        Gradients = []
        Loss = []
        for i in range(FLAGS.num_epochs):
            _, gradients, loss = sess.run([train_op, gradients_node, loss_op],
                                          feed_dict={x_input: x_train, 
                                                    y_input: y_train})
            print_list = [i+1, loss, gradients[0]]
            print("epoch: {0[0]} \t loss: {0[1]:.2f} \t gradients: {0[2]:.2f}".format(print_list))
            Gradients.append(gradients)
            Loss.append(loss)
    sess.close()

fig = plt.figure(1, (14, 6))
AX = [fig.add_subplot(i) for i in range(121,123)]
name = ['Loss', 'Gradients']
color = ['r', 'b']
data = [Loss, Gradients]
for na, ax, co, d in zip(name, AX, color, data):
  ax.plot(np.linspace(0, FLAGS.num_epochs, FLAGS.num_epochs), d, co, label=na)
  ax.set_title(na)
  ax.legend()

plt.show()

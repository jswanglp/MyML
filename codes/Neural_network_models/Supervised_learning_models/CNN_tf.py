# 该程序通过 TensorFlow 搭建两层卷积神经网络实现对 mnist 数据集的分类
# coding: utf-8

import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# 定义初始化函数
def glorot_init(shape, name):
    initial = tf.truncated_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))
    return tf.Variable(initial, name=name)

def bias_init(shape, name):
    initial =  tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':

    # 设置超参及路径
    mnist = input_data.read_data_sets('sample_data/MNIST_data', one_hot=True)
    tf.app.flags.DEFINE_integer('num_epochs', 12000, 'number of epochs, default is 12000.')
    tf.app.flags.DEFINE_integer('batch_size', 196, 'batchsize, default is 196.') # 小显存的噩梦
    tf.app.flags.DEFINE_float('learning_rate', 8e-4, 'learning rate, default is 8e-4.')

    FLAGS = tf.app.flags.FLAGS

    dir_path = os.path.dirname(os.path.abspath(__file__))
    event_path = os.path.join(dir_path, 'Tensorboard')
    checkpoint_path = os.path.join(dir_path, 'Checkpoints')

    # 设置网络图
    graph = tf.Graph()
    with graph.as_default():

        with tf.name_scope('Input'):
            x = tf.placeholder(tf.float32, shape=[None, 784], name='input_images')
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            keep_prob = tf.placeholder(tf.float32)


        # --------------conv1-----------------------------------
        with tf.name_scope('Conv1'):
            with tf.name_scope('weights_conv1'):
                W_conv1 = glorot_init([3, 3, 1, 64], 'w_conv1') # 小显存的噩梦
            with tf.name_scope('bias_covn1'):
                b_conv1 = bias_init([64], 'b_conv1')            
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            with tf.name_scope('features_conv1'):
                h_pool1 = max_pool_2x2(h_conv1)

        # --------------conv2-----------------------------------
        with tf.name_scope('Conv2'):
            W_conv2 = glorot_init([3, 3, 64, 128], 'w_conv2') # 小显存的噩梦
            b_conv2 = bias_init([128], 'b_conv2')
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        # --------------fc--------------------------------------
        h_pool2_flat = tf.layers.flatten(h_pool2)
        num_f = h_pool2_flat.get_shape().as_list()[-1]
        with tf.name_scope('FC1'):
            W_fc1 = glorot_init([num_f, 128], 'w_fc1')
            b_fc1 = bias_init([128], 'b_fc1')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        with tf.name_scope('Dropout'):    
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            
        with tf.name_scope('FC2'):
            W_fc2 = glorot_init([128, 10], 'w_fc2')
            b_fc2 = bias_init([10], 'b_fc2')
            y_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
        with tf.name_scope('Loss'):
            y_out = tf.nn.softmax(y_fc2)
            # cross_entropy = -tf.reduce_mean(y_*tf.log(y_out + 1e-10))
            # # or like
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,
                                                                                    logits=y_fc2))
        with tf.name_scope('Train'):
            train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cross_entropy)
            # # or like
            # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            # grad_list = optimizer.compute_gradients(cross_entropy)
            # train_step = optimizer.apply_gradients(grad_list)

        with tf.name_scope('Accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    # 训练并保存网络
    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3) # 定义保存3个模型
        max_acc = 101. # 超过该精度才会被保存
    
        for epoch in range(FLAGS.num_epochs):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            _, acc, loss = sess.run([train_step, accuracy, cross_entropy], feed_dict={x: batch[0], 
                                                                                    y_: batch[1], 
                                                                                    keep_prob: 0.5})
            step = epoch + 1
            if step % 1000 == 0:
                acc *= 100
                print_list = [step, loss, acc]
                print("Epoch: {0[0]}, cross_entropy: {0[1]:.4f}, accuracy on training data: {0[2]:.2f}%.".format(print_list))
                test_acc, test_loss = sess.run([accuracy, cross_entropy], feed_dict={x: mnist.test.images, 
                                                                                    y_: mnist.test.labels, 
                                                                                    keep_prob: 1.0})
                test_acc *= 100
                print_list = [test_loss, test_acc]
                print(' '*12, 'cross_entropy: {0[0]:.4f}, accuracy on testing data: {0[1]:.2f}%.'.format(print_list))
                print('\n')
            
            if (acc > max_acc) & (step > 3999): # 保存精度高的三个模型
                max_acc = acc
                saver.save(sess, os.path.join(checkpoint_path, 'f_map.ckpt'), global_step=step)

        test_image, test_label = mnist.test.images[100, :].reshape((1, -1)), mnist.test.labels[100, :].reshape((1, -1))
        features1, features2 = sess.run([h_pool1, h_pool2], feed_dict={x: test_image, y_: test_label, keep_prob: 1.0})
    
    sess.close()
    
    # 复原保存的网络
    # with tf.Session() as sess:
        # model_path = 'GoogleDrive/My Drive/Colab Notebooks/Tensorboard/f_map.ckpt-241'
        # saver.restore(sess, model_path)
        # acc, loss = sess.run([accuracy, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        # print('Accuracy is %.2f.' %(acc))
    # sess.close()
    
    # ----------------------显示 mnist.test.image 第100幅图像的第一层feature map(14*14*32)------------

    features_map = features1.reshape((14, 14, 64))
    num_map = range(features_map.shape[-1])
    fig, AX = plt.subplots(nrows=4, ncols=8)
    fig.set_size_inches(w=14, h=7)
    fig.subplots_adjust(wspace=.2, hspace=.2)
    try:
        for index, ax in enumerate(AX.flatten()):
            ax.imshow(features_map[:, :, index], 'gray')
            ax.set_xticks([]), ax.set_yticks([])
    except IndexError:
        pass
    # ----------------------显示 mnist.test.image 第100幅图像的第二层feature map(7*7*32)------------

    features_map = features2.reshape((7, 7, 128))
    num_map = range(features_map.shape[-1])
    fig, AX = plt.subplots(nrows=4, ncols=8)
    fig.set_size_inches(w=14, h=7)
    fig.subplots_adjust(wspace=.2, hspace=.2)
    try:
        for index, ax in enumerate(AX.flatten()):
            ax.imshow(features_map[:, :, index], 'gray')
            ax.set_xticks([]), ax.set_yticks([])
    except IndexError:
        pass
    plt.show()
    
    

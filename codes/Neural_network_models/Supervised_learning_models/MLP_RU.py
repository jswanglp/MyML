# В программе реализована классификация набора изображений mnist с помощью трехслойного персептрона
# Определение функции printRed, которая может отображать красные шрифты в командной строке
# Определение функции print_progress, которая отображает ход выполнения в реальном времени
# Использование метода mini-batch SGD для обучения персептрона
# -*- coding: utf-8 -*- 
"""
Created on 06 May, 2019
@author jswanglp

requirements:
    Keras==2.2.4
    matplotlib==2.0.2
    numpy==1.15.4
    tensorflow==1.12.0
    scipy==1.1.0
    Bunch==1.0.1

"""

import numpy as np
import os, sys, ctypes
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from progress_bar import print_progress

# -----------------------Определение функции printRed()---------------------------
STD_INPUT_HANDLE = -10
STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12

FOREGROUND_RED = 0x0c # red.
FOREGROUND_BLUE = 0x09 # blue.
FOREGROUND_GREEN = 0x0a # green.

std_out_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)

def set_cmd_text_color(color, handle=std_out_handle):
    Bool = ctypes.windll.kernel32.SetConsoleTextAttribute(handle, color)
    return Bool

def resetColor():
    set_cmd_text_color(FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE)

def printRed(mess):
    set_cmd_text_color(FOREGROUND_RED)
    sys.stdout.write(mess)
    resetColor()

# printRed("Model restored ... ...\n") # тест
# print("Model restored ... ...\n")
#--------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    # Настройка парамнтров 
    tf.app.flags.DEFINE_string('events_path', os.path.dirname(os.path.abspath(__file__)) + '/Tensorboard', 'Directory where event logs are written to.')
    tf.app.flags.DEFINE_string('checkpoints_path', os.path.dirname(os.path.abspath(__file__)) + '/Checkpoints', 'Directory where checkpoints are written to.')
    
    tf.app.flags.DEFINE_integer('max_num_checkpoints', 3, 'Maximum number of checkpoints that TensorFlow will keep.')
    tf.app.flags.DEFINE_integer('num_class', 10, 'Number of classes.')
    tf.app.flags.DEFINE_integer('batch_size', 256, 'Number of model batchsize.')
    tf.app.flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs for training.')
    tf.app.flags.DEFINE_integer('num_neurons', 250, 'Number of neurons in hidden layers.')
    tf.app.flags.DEFINE_integer('num_batch', 100, 'Number of batchs in one epoch.')

    tf.app.flags.DEFINE_float('learning_rate', 5e-3, 'Initial learning rate.')
    tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')
    tf.app.flags.DEFINE_float('decay_steps', 16., 'Decay steps')
    tf.app.flags.DEFINE_float('keep_prob', 0.5, 'The probability that each element is kept.')
    
    tf.app.flags.DEFINE_boolean('online_test', True, 'Online test or not ?')

    FLAGS = tf.app.flags.FLAGS

    if not os.path.isabs(FLAGS.events_path):
        raise ValueError('You must assign absolute path for --events_path')

    if not os.path.isabs(FLAGS.checkpoints_path):
        raise ValueError('You must assign absolute path for --checkpoints_path')

    # Выделение обучающих и тестовых изображений
    mnist = input_data.read_data_sets("sample_data/MNIST_data", reshape=True, one_hot=True)
    train_image = mnist.train.images
    train_label = mnist.train.labels
    test_image = mnist.test.images
    test_label = mnist.test.labels

    num_samples, num_features = train_image.shape

    # Создание graph
    graph = tf.Graph()
    with graph.as_default():

        # learning rate：        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        decay_steps = FLAGS.decay_steps
        learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=decay_steps,
                                                    decay_rate=FLAGS.learning_rate_decay_factor,
                                                    staircase=True,
                                                    name='exponential_decay')
        
        # Определения placeholder
        image_place = tf.placeholder(tf.float32, shape=[None, num_features], name='image')
        label_place = tf.placeholder(tf.float32, shape=[None, FLAGS.num_class], name='label')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Сетевая структура трехслойного персептрона
        #-------------------fc-1--------------------
        output_fc1 = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=FLAGS.num_neurons, scope='fc-1')
        #-------------------fc-2--------------------
        output_fc2 = tf.contrib.layers.fully_connected(inputs=output_fc1, num_outputs=FLAGS.num_neurons, scope='fc-2')
        #-----------------dropout layer-------------
        output_dp = tf.contrib.layers.dropout(inputs=output_fc2, keep_prob=keep_prob, scope='dropout-layer')
        #-------------------fc-3--------------------
        output_pre_softmax = tf.contrib.layers.fully_connected(inputs=output_dp, num_outputs=FLAGS.num_class, scope='fc-3')

        # Определение loss
        with tf.name_scope('loss'):
            loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_pre_softmax,
                                                                                labels=label_place, name='loss_tensor'))

        # Определение accuracy
        with tf.name_scope('accuracy'):
            prediction = tf.equal(tf.arg_max(output_pre_softmax, 1), tf.arg_max(label_place, 1))
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        # Два способа определения optimizer
        # with tf.name_scope('train'):
        #     train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_tensor, global_step=global_step)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gradients_vars = optimizer.compute_gradients(loss_tensor)
            train_op = optimizer.apply_gradients(gradients_vars, global_step=global_step)
        
        # 定义 summaries
        tf.summary.scalar('loss', loss_tensor, collections=['train', 'test'])
        tf.summary.scalar('accuracy', accuracy, collections=['train', 'test'])
        tf.summary.scalar('global_step', global_step, collections=['train'])
        tf.summary.scalar('learning_rate', learning_rate, collections=['train'])

        summary_train = tf.summary.merge_all('train')
        summary_test = tf.summary.merge_all('test')
        
        max_acc = 99.0 # модели с выше этой точностью будут сохранены
        min_cross = 0.1
        # if not os.path.exists(FLAGS.checkpoints_path):
        #     os.makedirs(FLAGS.checkpoints_path)

        # Обучение персептрона
        sess = tf.Session(graph=graph)
        with sess.as_default():
            # Определения saver, summary writer
            saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoints)
            sess.run(tf.global_variables_initializer())

            train_summary_dir = os.path.join(FLAGS.events_path, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir)
            train_summary_writer.add_graph(sess.graph)

            test_summary_dir = os.path.join(FLAGS.events_path, 'summaries', 'test')
            test_summary_writer = tf.summary.FileWriter(test_summary_dir)
            test_summary_writer.add_graph(sess.graph)

            for num_epoch in range(FLAGS.num_epochs):

                image_batch, label_batch = mnist.train.next_batch(FLAGS.batch_size)
                batch_loss, batch_acc, _, batch_summ, num_train, lr = sess.run([loss_tensor, accuracy, train_op, summary_train, global_step, learning_rate],
                                                                            feed_dict={image_place: image_batch, label_place: label_batch,
                                                                                        keep_prob: FLAGS.keep_prob})
                batch_acc *= 100
                progress = float(num_epoch % FLAGS.num_batch + 1) / FLAGS.num_batch
                num_epoch_batch = num_epoch // FLAGS.num_batch + 1
                print_progress(progress, num_epoch_batch, batch_loss, batch_acc)
                # print('Epoch '+str(num_epoch+1) + ', learning rate is %.4f' % lr + ', train accuracy is '+'{:.2f}%.'.format(batch_acc))
                # print('Epoch '+str(num_epoch+1) + ', learning rate is %.4f' % lr + ', train loss is '+ '{:.4f}, '.format(batch_loss) + \
                #         'accuracy is '+'{:.2f}%.'.format(batch_acc))
                train_summary_writer.add_summary(batch_summ, num_epoch)

                checkpoints_prefix = 'model.ckpt'
                if (num_epoch+1) % FLAGS.num_batch == 0:
                    print('Learning rate is %.4f' % lr + ', train accuracy is '+'{:.2f}%.'.format(batch_acc))
                    if (batch_loss <= min_cross) & (batch_acc > max_acc):
                        min_cross = batch_loss
                        max_acc = batch_acc
                        saver.save(sess, os.path.join(FLAGS.checkpoints_path, checkpoints_prefix), global_step=num_epoch+1)
                        # print("\033[0;31;40m\tModel restored ... ...\033[0m\n")
                        printRed("Model restored ... ...\n")
                        print('\n')

                    if FLAGS.online_test:
                        test_loss, test_acc, test_summ = sess.run([loss_tensor, accuracy, summary_test],
                                                                    feed_dict={image_place: test_image, label_place: test_label,
                                                                    keep_prob: 1.})
                        test_acc *= 100
                        print('Test accuracy is '+'{:.2f}%.\n'.format(test_acc))
                        test_summary_writer.add_summary(test_summ, num_epoch)
            
            # Точность в наборе тестовых данных
            test_accuracy = sess.run([accuracy], feed_dict={image_place: test_image, label_place: test_label, 
                                                            keep_prob: 1.})
            test_acc = test_accuracy[0]*100
            print("Final Test Accuracy is %.2f%%" % test_acc)

            train_summary_writer.close()
            test_summary_writer.close()
        sess.close()

# См. Logistic_regression.py на код для восстановления сохраненной модели

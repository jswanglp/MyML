# 该程序用 Softmax 回归解决二分类问题
# 改写自 logistic regression 程序，默认采用 SGD 来训练网络
# #@title Softmax Regression { display-mode: "both" }
# coding:utf-8
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)

# 只获取0或1的图像的索引
def extraction_fn(data):
    index_list = []
    for idx in range(data.shape[0]):
        if data[idx] == 0 or data[idx] == 1:
            index_list.append(idx)
    return index_list

# Xavier Glorot 参数初始化 
def glorot_init(shape, name):
    initial = tf.truncated_normal(shape, stddev=1. / tf.sqrt(shape[0] / 2.))
    return tf.Variable(initial, name=name)

if __name__ == '__main__':

    print('All the following parameters can be customized by adding instructions: --name definition') # 声明可自定义的参数
    print('1. envents_path, '+'2. checkpoints_path, '+'3. max_num_checkpoints, '\
        +'4. num_classes, '+'5. batchsize, '+ '6. num_epochs, '+'7. learning_rate.\n')

    tf.app.flags.DEFINE_string('events_path', os.path.dirname(os.path.abspath(__file__)) + '/Tensorboard', 'Where events are writen to.')
    tf.app.flags.DEFINE_string('checkpoints_path', os.path.dirname(os.path.abspath(__file__)) + '/Checkpoints', 'Where checkpoints are writen to.')

    tf.app.flags.DEFINE_integer('max_num_checkpoints', 3, 'Maximum number of checkpoints that TensorFlow will keep.')
    tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of model clones to deploy.')
    tf.app.flags.DEFINE_integer('batchsize', 64, 'Number of batchsize.')
    tf.app.flags.DEFINE_integer('num_epochs', 10000, 'Number of epochs for training.')
    tf.app.flags.DEFINE_integer('display_step', 1000, 'Display step for showing loss and accuracy.')
    
    tf.app.flags.DEFINE_float('learning_rate', 5e-4, 'Initial learning rate.')
    
    # tf.app.flags.DEFINE_boolean('')

    FLAGS = tf.app.flags.FLAGS # 收集所有的FLAGS
    print('The events_path is ', FLAGS.events_path)
    print('The checkpoints_path is ', FLAGS.checkpoints_path)

    # 所需训练图像和标签的获取
    mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
    data = {}

    index_list_train = extraction_fn(mnist.train.labels)
    index_list_test = extraction_fn(mnist.test.labels)

    data['train_imgs'], data['train_lbs'] = mnist.train.images[index_list_train], mnist.train.labels[index_list_train]
    data['test_imgs'], data['test_lbs'] = mnist.test.images[index_list_test], mnist.test.labels[index_list_test]

    data['train_imgs_lbs'] = np.c_[data['train_imgs'], data['train_lbs']]
    num_samples, num_features = data['train_imgs'].shape

    # 网络的设置
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('Main_structure'):
            with tf.name_scope('Inputs'):
                image_place = tf.placeholder(tf.float32, shape=[None, num_features], name='images')
                label_place = tf.placeholder(tf.int32, shape=[None,], name='labels')
                label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)

                w = glorot_init([num_features, FLAGS.num_classes], name='weights')
                b = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_classes]), name='biases')
                logits = tf.add(tf.matmul(image_place, w), b)

            with tf.name_scope('Loss'):
                loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot), name='cross_entropy')

            with tf.name_scope('Accuracy'):
                predition = tf.equal(tf.argmax(logits, 1), tf.arg_max(label_one_hot, 1))
                accuracy = tf.reduce_mean(tf.cast(predition, tf.float32), name='accuracy')

            with tf.name_scope('Train'):
                train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss_tensor)
                
        image_sum = tf.summary.image('input_images', tf.reshape(image_place, [-1, 28, 28, 1]), 
                                    max_outputs=3, collections=['train']) #选取3个输入图像展示
        loss_sum = tf.summary.scalar('loss_summary', loss_tensor, ['train', 'test']) # 关于loss的summary
        acc_sum = tf.summary.scalar('acc_summary', accuracy, ['test']) # 关于accuracy的summary

        train_summ = tf.summary.merge_all('train')
        test_summ = tf.summary.merge_all('test')

        saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoints)
    
    max_acc = 99.2 # 高于此精度的模型将被saved
    min_cross = 0.2
    if not os.path.exists(FLAGS.checkpoints_path):
        os.makedirs(FLAGS.checkpoints_path)
    if not os.path.exists(FLAGS.events_path):
        os.makedirs(FLAGS.events_path)

    # 模型的训练
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        checkpoints_prefix = 'model.ckpt'

        train_dir = os.path.join(FLAGS.events_path, 'train')
        test_dir = os.path.join(FLAGS.events_path, 'test')
        train_writer = tf.summary.FileWriter(train_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_dir, sess.graph)

        # # Mini-batch SGD
        # for epoch_num in range(FLAGS.num_epochs):
        #     num_batches = num_samples // FLAGS.batchsize
        #     np.random.shuffle(data['train_imgs_lbs'])
        #     for batch_num in range(num_batches):
        #         index_list_start = batch_num * FLAGS.batchsize
        #         index_list_end = (batch_num + 1) * FLAGS.batchsize
        #         image_batch = data['train_imgs_lbs'][index_list_start:index_list_end, :-1]
        #         label_batch = data['train_imgs_lbs'][index_list_start:index_list_end, -1]
        #         batch_loss, batch_accuracy, _ = sess.run([loss_tensor, accuracy, train_op],
        #                                                 feed_dict={image_place: image_batch, label_place: label_batch})
        #     train_loss, train_acc, train_rs = sess.run([loss_tensor, accuracy, train_summ],
        #                                                 feed_dict={image_place: data['train_imgs'], label_place: data['train_lbs']})
        
        # SGD -------------------------------------------------------------------------------------------------------------------
        for epoch_num in range(FLAGS.num_epochs):
            img_index = np.random.randint(0, num_samples, 1)[0]
            image_and_label = data['train_imgs_lbs'][img_index]
            image = image_and_label[:-1].reshape([-1, num_features])
            label = np.array(image_and_label[-1]).reshape([1, ])
            _, train_loss, train_acc, train_rs = sess.run([train_op, loss_tensor, accuracy, train_summ],
                                                        feed_dict={image_place: image, label_place: label})
        # -----------------------------------------------------------------------------------------------------------------------
            
            train_writer.add_summary(train_rs, global_step=epoch_num)
            test_loss, test_acc, test_rs = sess.run([loss_tensor, accuracy, test_summ],
                                                        feed_dict={image_place: data['test_imgs'], label_place: data['test_lbs']})
            test_writer.add_summary(test_rs, global_step=epoch_num)

            if (epoch_num + 1) % FLAGS.display_step == 0 or epoch_num == 0:
                train_acc *= 100
                test_acc *= 100
                print("Epoch " + str(epoch_num + 1) + ", Cross_entropy loss is " + \
                      "{:.5f}, ".format(train_loss) + "accuracy is " + "{:.2f}%".format(train_acc))
                print("Cross_entropy loss on the whole testing set is " + \
                      "{:.5f}, ".format(test_loss) + "accuracy is " + "{:.2f}%\n".format(test_acc))

            if (test_loss <= min_cross) & (test_acc > max_acc): # 按照要求保存网络模型
                min_cross = test_loss
                max_acc = test_acc
                saver.save(sess, os.path.join(FLAGS.checkpoints_path, checkpoints_prefix), global_step=epoch_num+1)
                print("Model restored...") 

    train_writer.close()
    test_writer.close()
    sess.close()

# # 恢复指定的网络并测试新的数据
    # with tf.Session() as sess:
    #     # checkpoints_prefix = 'model.ckpt'
    #     ckpt_path = os.path.join(FLAGS.checkpoints_path, checkpoints_prefix)+'-146'
    #     saver.restore(sess, ckpt_path)
    #     # saver = tf.train.import_meta_graph(meta_path)
    #     # saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoints_path))
    #     test_loss, test_accuracy = sess.run([loss_tensor, accuracy],
    #                                     feed_dict={image_place: data['test_image'],
    #                                                 label_place: data['test_label'],
    #                                                 keep_prob: 1.})
    #     test_accuracy *= 100
    #     print('The loss is: {:.4f}, '.format(test_loss)+'and the test accuracy is %.2f.' % test_accuracy)
    # sess.close()

# 该程序用单层全连接神经网络和 softmax 层实现逻辑回归的二分类
# #@title Real Logistic Reression { display-mode: "both" }
# coding:utf-8
import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)


def extraction_fn(data): # 只获取0或1的图像的索引
    index_list = []
    for idx in range(data.shape[0]):
        if data[idx] == 0 or data[idx] == 1:
            index_list.append(idx)
    return index_list

if __name__ == '__main__':

    print('All the following parameters can be customized by adding instructions: --name definition') # 声明可自定义的参数
    print('1. envents_path, '+'2. checkpoints_path, '+'3. max_num_checkpoints, '\
        +'4. num_classes, '+'5. batchsize, '+ '6. num_epochs, '+'7. learning_rate.\n')

    tf.app.flags.DEFINE_string('events_path', os.path.dirname(os.path.abspath(__file__)) + '/Tensorboard', 'Where events are writen to.')
    tf.app.flags.DEFINE_string('checkpoints_path', os.path.dirname(os.path.abspath(__file__)) + '/Checkpoints', 'Where checkpoints are writen to.')

    tf.app.flags.DEFINE_integer('max_num_checkpoints', 3, 'Maximum number of checkpoints that TensorFlow will keep.')
    tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of model clones to deploy.')
    tf.app.flags.DEFINE_integer('batchsize', 64, 'Number of batchsize.')
    tf.app.flags.DEFINE_integer('num_epochs', 100, 'Number of epochs for training.')

    tf.app.flags.DEFINE_float('learning_rate', 5e-3, 'Initial learning rate.')
    
    # tf.app.flags.DEFINE_boolean('')

    FLAGS = tf.app.flags.FLAGS # 收集所有的FLAGS
    print('The events_path is ', FLAGS.events_path)
    print('The checkpoints_path is ', FLAGS.checkpoints_path)

# 所需训练图像和标签的获取
    mnist = input_data.read_data_sets("E:\Anaconda\Programs\MNIST_data", reshape=True, one_hot=False)
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
    data['test_label'] = mnist.test.labels[index_list_test]

    data['train_image_label'] = np.c_[data['train_image'], data['train_label']]
    num_samples, num_features = data['train_image'].shape

# 网络的设置
    with tf.name_scope('Inputs'):
        image_place = tf.placeholder(tf.float32, shape=[None, num_features], name='images')
        image_sum = tf.summary.image('input_images', tf.reshape(image_place, [-1, 28, 28, 1]), max_outputs=3) #选取3个输入图像展示
        label_place = tf.placeholder(tf.int32, shape=[None,], name='labels')
        label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
    
    with tf.name_scope('Loss'):
        logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=FLAGS.num_classes, scope='fc')
        loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot), name='loss_tensor')
        loss_sum = tf.summary.scalar('loss_summary', loss_tensor) # 关于loss的summary
    
    with tf.name_scope('Accuracy'):
        predition = tf.equal(tf.argmax(logits, 1), tf.arg_max(label_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(predition, tf.float32), name='accuracy')
        acc_sum = tf.summary.scalar('acc_summary', accuracy) # 关于accuracy的summary

    with tf.name_scope('Train'):
        train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss_tensor)

    saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoints)
    max_acc = 99.2 # 高于此精度的模型将被saved
    min_cross = 0.2
    if not os.path.exists(FLAGS.checkpoints_path):
        os.makedirs(FLAGS.checkpoints_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoints_prefix = 'model.ckpt'

        writer = tf.summary.FileWriter(FLAGS.events_path, sess.graph)
        merged = tf.summary.merge([image_sum, loss_sum, acc_sum])

        for epoch_num in range(FLAGS.num_epochs):
            # 该段用于整个batch的按序提取及训练，速度较慢，梯度波动较小
            # num_batches = int(num_samples/FLAGS.batchsize)
            # for batch_num in range(num_batches):
            #     index_list_start = batch_num*FLAGS.batchsize
            #     index_list_end = (batch_num+1)*FLAGS.batchsize
            #     image_batch = data['train_image'][index_list_start:index_list_end,:]
            #     label_batch = data['train_label'][index_list_start:index_list_end]
            #     batch_loss, batch_accuracy, _ = sess.run([loss_tensor, accuracy, train_op],
            #                                             feed_dict={image_place: image_batch, label_place: label_batch})
            #该段用于随机batch的训练，速度较快，梯度波动较大
            #--------------------------------------------------------------------------------------------------------
            np.random.shuffle(data['train_image_label'])
            image_batch = data['train_image_label'][:FLAGS.batchsize,:-1]
            label_batch = data['train_image_label'][:FLAGS.batchsize,-1]
            #--------------------------------------------------------------------------------------------------------

            batch_loss, batch_accuracy, _ = sess.run([loss_tensor, accuracy, train_op],
                                                    feed_dict={image_place: image_batch, label_place: label_batch})
            if (epoch_num + 1) % 5 == 0 or (epoch_num + 1) == 1:
                batch_accuracy *= 100
                print("Epoch " + str(epoch_num + 1) + ", Training Loss is " + \
                      "{:.5f}, ".format(batch_loss) + "batch_accuracy is " + "{:.2f}%".format(batch_accuracy))

            if (batch_loss <= min_cross) & (batch_accuracy > max_acc): # 按照要求保存网络模型
                min_cross = batch_loss
                max_acc = batch_accuracy
                saver.save(sess, os.path.join(FLAGS.checkpoints_path, checkpoints_prefix), global_step=epoch_num+1)
                print("Model restored...")
            rs = sess.run(merged, feed_dict={image_place: image_batch, label_place: label_batch})
            writer.add_summary(rs, epoch_num)
        
        # 测试集精度
        test_accuracy = sess.run([accuracy], feed_dict={image_place: data['test_image'],
                                                        label_place: data['test_label']})
        test_acc = test_accuracy[0]*100
        print("Final Test Accuracy is %.2f%%" % test_acc)
    writer.close()
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

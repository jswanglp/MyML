# VGG16的调用
# 原始的 repo 在这里: https://github.com/machrisaa/tensorflow-vgg
# 训练好的 vgg16 模型参数在这里: https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
# 需要的文件包括: vgg16.py, utils.py, synset.txt, vgg16.npy, test_data
# 修改了 utils.py 中 print_prob函数，返回名称和概率
# 修改了 vgg16.py 中 __init__ 函数 allow_pickle=True
# 添加了一次识别多张图像并且输出结果的程序
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
import sys,os
import time

# tf.logging.set_verbosity(tf.logging.ERROR)

# 切换工作路径
file_path = '/content/GoogleDrive/My Drive/Colab Notebooks/vgg16'
os.chdir(file_path)
os.getcwd()

# Import 主函数
import vgg16,utils

if __name__ == '__main__':
    
    # 获取所有测试图像
    im_path = os.path.join(file_path, 'test_data')
    im_list = [os.path.join(im_path, f) for f in os.listdir(im_path)]
    print(im_list)

    # 识别所有图片，返回类别名称及概率
    NAME, PROB = [], []
    with tf.Session() as sess:
        vgg = vgg16.Vgg16()
        with tf.name_scope('Input'):
            image = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name='input_image')
            vgg.build(image)
        start_time = time.time()
        for im_l in im_list:
            img = utils.load_image(im_l)
            prob = sess.run(vgg.prob, feed_dict={image: img.reshape((1, 224, 224, 3))})
    #         print(prob)
            name, p = utils.print_prob(prob.flatten(), './synset.txt')
            NAME.append(name)
            PROB.append(p)
        end_time = time.time()

    # 提取所有类别名称，概率
    num = len(im_list)
    print('There are {} images in total, recognition time: {:.4f}s.'.format(num, end_time - start_time))
    TITLE = [', '.join([name[10:], str(p)]) for name, p in zip(NAME, PROB)]
    print(TITLE)

    # 打印所有图片及其所属类别、概率
    cols = 3 # 每行显示的图片数
    nrows = num // cols + 1
    height = nrows * 5
    fig = plt.figure(1, figsize=(15, height))
    gs = gridspec.GridSpec(nrows, 3)
    for i, im_l in enumerate(im_list):
        row = i // cols
        col = i % cols
        img = utils.load_image(im_l)
        img = img / img.max()
        title = TITLE[i]
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(title.capitalize(), fontsize=14)
        ax.set_xticks([]), ax.set_yticks([])

    # fig.tight_layout()
    plt.show()

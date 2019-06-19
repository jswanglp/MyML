# Применение VGG сетей
# Оригинальный repo здесь: https://github.com/machrisaa/tensorflow-vgg
# Параметры обученной модели vgg16 здесь: https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
# Необходимые документы включают: vgg16.py, utils.py, synset.txt, vgg16.npy, test_data
# Модифицирована функция print_prob в utils.py, теперь она возвращает название категории и вероятность
# Модифицирована функция __init__ в vgg16.py, и параметр allow_pickle установлен True
# Добавлен код, который распознает несколько изображений одновременно и выводит результаты

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
import sys,os
import time

# tf.logging.set_verbosity(tf.logging.ERROR)

# Переключение work_path
file_path = '/content/GoogleDrive/My Drive/Colab Notebooks/vgg16'
os.chdir(file_path)
os.getcwd()

# Import главные функции
import vgg16,utils

if __name__ == '__main__':
    
    # Извлечение путей всех тестовых изображений
    im_path = os.path.join(file_path, 'test_data')
    im_list = [os.path.join(im_path, f) for f in os.listdir(im_path)]
    print(im_list)

    # Распознавание всех изображений и возвращение названий категорий и вероятностей
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

    # Извлечение названий категорий и вероятностей всех ихображений
    num = len(im_list)
    print('There are {} images in total, recognition time: {:.4f}s.'.format(num, end_time - start_time))
    TITLE = [', '.join([name[10:], str(p)]) for name, p in zip(NAME, PROB)]
    print(TITLE)

    # Представление всех изображений и их категорий, вероятностей
    cols = 3 # количество картинок в строке
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

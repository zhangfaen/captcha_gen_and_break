import random
import sys

import numpy as np
from PIL import Image
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Dense, Flatten, Dropout
from keras.models import Model
from keras.utils import np_utils
from keras.utils.visualize_util import plot

CLASS_NUM = 10


def one_hot_encode(label):
    return np_utils.to_categorical(np.int32(list(label)), CLASS_NUM)


def load_data(path, train_ratio, _shape):
    datas = []
    labels = []
    input_file = open(path + '/labels.txt')
    height = _shape[0]
    width = _shape[1]
    for i, line in enumerate(input_file):
        img = Image.open(path + '/' + str(i) + ".png")
        data = img.resize([width, height])
        data = np.multiply(data, 1 / 255.0)
        data = np.asarray(data)
        datas.append(data)
        labels.append(one_hot_encode(line.strip()))
    input_file.close()

    datas_labels = zip(datas, labels)
    random.shuffle(datas_labels)
    (datas, labels) = zip(*datas_labels)
    size = len(labels)
    train_size = int(size * train_ratio)
    train_datas = np.stack(datas[0: train_size])
    train_labels = np.stack(labels[0: train_size])
    test_datas = np.stack(datas[train_size: size])
    test_labels = np.stack(labels[train_size: size])
    return (train_datas, train_labels, test_datas, test_labels)


def get_cnn_net(num, _shape):
    inputs = Input(shape=_shape)
    conv1 = Convolution2D(32, 5, 5, border_mode='same', activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = Convolution2D(32, 5, 5, activation='relu')(drop1)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.15)(pool2)

    conv3 = Convolution2D(32, 3, 3, activation='relu')(drop2)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.15)(pool3)

    flat = Flatten()(drop3)
    y_list = []
    for i in xrange(num):
        y_list.append(Dense(CLASS_NUM, activation='softmax')(flat))
    model = Model(input=inputs, output=y_list)
    plot(model, show_shapes=True, to_file='cnn_four_vectors.png')
    model.compile(loss='categorical_crossentropy',
                  loss_weights=[1.] * num,
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model


def evaluate(model, test_datas, test_labels, batch_size):
    metrics = {}
    metrics['accs'] = []
    predict_labels = model.predict(test_datas, batch_size)
    _shape = test_labels.shape
    acc_all = np.asarray([True] * _shape[0])
    for i in xrange(_shape[1]):
        acc_i = test_labels[:, i, :].argmax(1) == predict_labels[i].argmax(1)
        acc_all = acc_all * acc_i
        metrics['accs'].append(acc_i.mean())
    metrics['acc'] = acc_all.mean()
    return metrics


data_path = '../gen/captcha_data4/'
nb_epoch = 20
# data_path = sys.argv[1]
# nb_epoch = int(sys.argv[2])
num_figure = 4
width = 160
height = 60
channel = 3
shape = (height, width, channel)
batch_size = 32
train_ratio = 0.9
(train_datas, train_labels, test_datas, test_labels) = load_data(data_path, train_ratio, shape)
model = get_cnn_net(num_figure, shape)
t0 = time.time()
model.fit(train_datas, list(np.transpose(train_labels, (1, 0, 2))), batch_size, nb_epoch)
t1 = time.time()
print t1 - t0
json_string = model.to_json()
open('cnn_four_vectors.json','w').write(json_string)
model.save_weights('cnn_four_vectors.h5')

print "train evaluations:"
model_metrics = model.evaluate(train_datas, list(np.transpose(train_labels, (1, 0, 2))), batch_size)
print dict(zip(model.metrics_names, model_metrics))
print evaluate(model, train_datas, train_labels, batch_size)
print "test evaluations:"
model_metrics = model.evaluate(test_datas, list(np.transpose(test_labels, (1, 0, 2))), batch_size)
print dict(zip(model.metrics_names, model_metrics))
print evaluate(model, test_datas, test_labels, batch_size)

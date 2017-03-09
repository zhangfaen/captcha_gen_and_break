from keras.layers import Input, Dense, Flatten, Activation
import keras.backend as K
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.utils.visualize_util import plot
import random
import numpy as np
import sys
from PIL import Image, ImageFilter

CLASS_NUM = 10


def one_hot_encode(label):
    return np.hstack(np_utils.to_categorical(np.int32(list(label)), CLASS_NUM))


def load_data(path, train_ratio, _shape):
    datas = []
    labels = []
    input_file = open(path + '/labels.txt')
    height = _shape[0]
    width = _shape[1]
    for i, line in enumerate(input_file):
        img = Image.open(path + '/' + str(i) + ".png")
        img = img.resize([width, height])
        img = img.filter(ImageFilter.MaxFilter).filter(ImageFilter.MinFilter)
        img = img.convert('L')
        data = np.asarray(img)
        data = np.divide(data, int(np.mean(data)))
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


def one_vector_metrics(y_true, y_pred):
    num = 4
    metrics = {}
    _y_true = K.reshape(y_true, (-1, 4, CLASS_NUM))
    _y_pred = K.reshape(y_pred, (-1, 4, CLASS_NUM))
    _y = K.equal(K.argmax(_y_true), K.argmax(_y_pred))

    y = K.equal(K.sum(K.cast(_y, 'int32'), -1), K.variable(num, 'int32'))
    metrics['acc'] = K.mean(y)

    yi = K.mean(_y, 0)
    for i in xrange(num):
        metrics['acc_' + str(i)] = yi[i]
    return metrics


def get_nn_net(num, _shape):
    inputs = Input(shape=_shape)
    flat = Flatten()(inputs)
    d1 = Dense(100, activation='relu')(flat)
    d2 = Dense(100, activation='relu')(d1)
    y = Dense(40, activation='softmax')(d2)
    model = Model(input=inputs, output=y)
    plot(model, show_shapes=True, to_file='dnn.png')
    model.compile(loss='categorical_crossentropy',
                  loss_weights=[1.],
                  optimizer='Adam',
                  metrics=[one_vector_metrics])
    return model


def evaluate(model, test_datas, test_labels, batch_size):
    metrics = {}
    metrics['accs'] = []
    predict_labels = model.predict(test_datas, batch_size)
    _shape = test_labels.shape
    acc_all = np.asarray([True] * _shape[0])
    for i in xrange(_shape[1] / CLASS_NUM):
        start = i * CLASS_NUM
        end = start + CLASS_NUM
        acc_i = test_labels[:, start:end].argmax(1) == predict_labels[:, start:end].argmax(1)
        acc_all = acc_all * acc_i
        metrics['accs'].append(acc_i.mean())
    metrics['acc'] = acc_all.mean()
    return metrics


data_path = '../gen/captcha4_level3/'
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

model = get_nn_net(num_figure, (height, width))
model.fit(train_datas, train_labels, batch_size, nb_epoch)

print "train evaluations:"
model_metrics = model.evaluate(train_datas, train_labels, batch_size)
print dict(zip(model.metrics_names, model_metrics))
print evaluate(model, train_datas, train_labels, batch_size)
print "test evaluations:"
model_metrics = model.evaluate(test_datas, test_labels, batch_size)
print dict(zip(model.metrics_names, model_metrics))
print evaluate(model, test_datas, test_labels, batch_size)

import random
import sys
import time

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

def one_hot_decode(label):
    return  label.argmax()

def load_data(path,  _shape):
    """
        Load images and load corresponding labels.
        image file name looks like 789.png, whose label is line 789 of labels.txt
    """
    datas = []
    labels = []
    input_file = open(path + '/labels.txt')
    # _shape (height,width,channel)
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

    # datas.shape : (size,height * width * channel)
    # labels : [label_0,label_1,label_2,label_3]
    # label.shape : list(size, CLASS_NUM)
    labels = list(np.transpose(np.stack(labels), (1, 0, 2)))
    datas = np.stack(datas)
    return (datas,labels)

def get_cnn_net(num, _shape):
    """
        Define the CNN network model.
        This model has 3 cov lays and 4 outputs.
        Each output is for one digit in the image.
        Finally we optimize the 4 output at the same time, each ouput contribute 1/4 the total loss.
    """
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
    plot(model, show_shapes=True, to_file='cnn.png')
    model.compile(loss='categorical_crossentropy',
                  loss_weights=[1.] * num,
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model


def evaluate(model, test_datas, test_labels):
    metrics = {}
    metrics['accs'] = []
    predict_labels = model.predict(test_datas)
    num_figure = len(test_labels)
    test_size = test_labels[0].shape[0]
    acc_all = 0
    acc_each = [0,0,0,0]
    for i in xrange(test_size):
        flags = True
        for j in xrange(num_figure):
            flag = one_hot_decode(test_labels[j][i]) == one_hot_decode(predict_labels[j][i])
            if(flag):
                acc_each[j] = acc_each[j] + 1
                flags = flags and flag
        if(flags):
            acc_all = acc_all + 1

    for j in xrange(num_figure):
        metrics['accs'].append(acc_each[j] * 1.0 /test_size)
    metrics['acc'] = acc_all * 1.0 / test_size
    return metrics

def save_model(model):
    json_string = model.to_json()
    model_file = open('cnn.json','w')
    model_file.write(json_string)
    model_file.close()
    model.save_weights('cnn.h5')
if __name__ == '__main__':
    data_path_prefix = '../gen/4'
    if(data_path_prefix.endswith('/')):
        data_path_prefix=data_path_prefix[:-1]
    train_data_path = data_path_prefix + '_train/'
    test_data_path = data_path_prefix + '_test/'
    nb_epoch = 10
    num_figure = 4
    width = 160
    height = 60
    channel = 3
    shape = (height, width, channel)
    (train_datas, train_labels) = load_data(train_data_path, shape)
    (test_datas, test_labels) = load_data(test_data_path, shape)
    model = get_cnn_net(num_figure, shape)
    t0 = time.time()
    model.fit(train_datas, train_labels, nb_epoch = nb_epoch)
    t1 = time.time()
    print 'training time : ' , t1 - t0
    save_model(model)

    print "train evaluations:"
    model_metrics = model.evaluate(train_datas, train_labels)
    print dict(zip(model.metrics_names, model_metrics))
    print evaluate(model, train_datas, train_labels)
    print "test evaluations:"
    model_metrics = model.evaluate(test_datas, test_labels)
    print dict(zip(model.metrics_names, model_metrics))
    print evaluate(model, test_datas, test_labels)

from keras.layers import Input, Dense, Flatten, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D,merge
import keras.metrics
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.utils.visualize_util import plot
import random
import numpy as np
from PIL import Image


NUM_OF_IMAGES = 5000
WIDTH = 160
HEIGHT = 60
CHANNEL = 3

def one_hot_encode (label) :
	return np_utils.to_categorical(np.int32(list(label)), 10)

def load_data(path,train_ratio) :
    datas = []
    labels = []
    input_file = open(path + '/labels.txt')
    for i,line in enumerate(input_file):
        img = Image.open(path + str(i) + ".png")
        data = img.resize([WIDTH,HEIGHT])
        data = np.multiply(data, 1/255.0)
        data = np.asarray(data)
        datas.append(data)
        labels.append(one_hot_encode(line.strip()))
    input_file.close()
    datas_labels = zip(datas,labels)
    random.shuffle(datas_labels)
    (datas,labels) = zip(*datas_labels)
    size = len(labels) 
    train_size = int(size * train_ratio)
    train_datas = np.stack(datas[ 0 : train_size ])
    train_labels = np.stack(labels[ 0 : train_size ])
    test_datas = np.stack(datas[ train_size : size ])
    test_labels = np.stack(labels[ train_size : size])
    return (train_datas,train_labels,test_datas,test_labels)
def get_cnn_net():	
    inputs = Input(shape=(HEIGHT, WIDTH, CHANNEL))
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(HEIGHT, WIDTH, CHANNEL)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Flatten())
    x = model(inputs)
    x1 = Dense(10, activation='softmax')(x)
    x2 = Dense(10, activation='softmax')(x)
    x3 = Dense(10, activation='softmax')(x)
    x4 = Dense(10, activation='softmax')(x)
    model = Model(input=inputs, output=[x1, x2, x3, x4])
    plot(model, show_shapes=True, to_file='simple_cnn_multi_output_model.png')
    model.compile(loss='categorical_crossentropy',
			  loss_weights=[1., 1., 1., 1.],
              optimizer='Adam',
              metrics=['accuracy'])
    return model


(train_datas,train_labels,test_datas,test_labels) = load_data('../capgen/pydata/',0.9)

model = get_cnn_net()
model.fit(train_datas, list(np.transpose(train_labels,(1,0,2))),
         nb_epoch=20, batch_size=32)

loss_and_metrics = model.evaluate(test_datas, list(np.transpose(test_labels,(1,0,2))), batch_size=32)
print  '\nmodel evaluate:\n','\n'.join(map(lambda x:str(x),zip(model.metrics_names,loss_and_metrics)))


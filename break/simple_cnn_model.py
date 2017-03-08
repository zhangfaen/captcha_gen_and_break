import numpy as np
from keras.layers import Convolution2D, MaxPooling2D, merge
from keras.layers import Input, Dense, Flatten, Activation, Dropout
from keras.models import Model
from keras.models import Sequential

datas = []
labels = []
str_labels = []

NUM_OF_IMAGES = 0
IMAGE_H = 0
IMAGE_W = 0
DIGITS = 4

with open("../capgen/data/data.csv") as f:
    header = f.readline().split(",")
    NUM_OF_IMAGES = int(header[0])
    IMAGE_H = int(header[1])
    IMAGE_W = int(header[2])

    for i in xrange(NUM_OF_IMAGES):
        str_label = f.readline().strip()
        str_labels.append(str_label)
        data = np.zeros((IMAGE_H, IMAGE_W, 1), dtype="float32")
        for r in xrange(IMAGE_H):
            cols = f.readline().split(",")
            for j in xrange(IMAGE_W):
                data[r, j, 0] = np.float32(cols[j])

        datas.append(data)
        label = np.zeros(DIGITS * 10, dtype="float32")
        index = 0
        for c in str_label:
            for i in xrange(0, 10):
                if c != str(i):
                    label[index] = 0
                else:
                    label[index] = 1.0 / 4
                index = index + 1
        # print cols[0], "  ", label
        labels.append(label)

train_datas = np.stack(datas[0:len(datas) * 9 / 10])
print train_datas.shape
train_labels = np.stack(labels[0:len(labels) * 9 / 10])
print train_labels.shape

test_datas = np.stack(datas[len(datas) * 9 / 10: len(datas)])
test_labels = np.stack(labels[len(labels) * 9 / 10: len(labels)])
test_str_labels = str_labels[len(str_labels) * 9 / 10: len(str_labels)]

inputs = Input(shape=(35, 80, 1))
model = Sequential()
model.add(Convolution2D(10, 3, 3, border_mode='valid', input_shape=(35, 80, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Convolution2D(10, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Flatten())
x = model(inputs)

x1 = Dense(10, activation='softmax')(x)
x2 = Dense(10, activation='softmax')(x)
x3 = Dense(10, activation='softmax')(x)
x4 = Dense(10, activation='softmax')(x)

x = merge([x1, x2, x3, x4], mode="concat")

output = Dense(40, activation='softmax')(x)

model = Model(input=inputs, output=output)

from keras.utils.visualize_util import plot

plot(model, show_shapes=True, to_file='simple_cnn_model.png')

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(train_datas, train_labels, nb_epoch=20, batch_size=32)

outputs = model.predict(test_datas)


def get_code_from_output(output):
    ret = ""
    for i in xrange(0, 4):
        maxV = 0
        maxI = 0
        for j in xrange(0, 9):
            if output[i * 10 + j] >= maxV:
                maxV = output[i * 10 + j]
                maxI = j
        ret += str(maxI)
    return ret


print len(outputs)

correct = 0
for i in xrange(len(outputs)):
    # print test_str_labels[i] + "," + get_code_from_output(outputs[i])
    if test_str_labels[i] == get_code_from_output(outputs[i]):
        correct = correct + 1

# total: 500, correct:108
print "total: " + str(len(outputs)) + ", correct:" + str(correct)

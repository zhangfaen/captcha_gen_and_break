from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Activation, Dropout
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, merge
from keras.utils import np_utils


import numpy as np

datas = []
labels_0 = []
labels_1 = []
labels_2 = []
labels_3 = []

str_labels = []

NUM_OF_IMAGES = 0
IMAGE_H = 0
IMAGE_W = 0
DIGITS = 4

def getLabelFromStr (digit) :
	return np_utils.to_categorical(int(digit), 10)[0]

with open("../capgen/data/data.csv") as f:
	header = f.readline().split(",")
	NUM_OF_IMAGES = int(header[0])
	IMAGE_H = int(header[1])
	IMAGE_W = int(header[2])

	for i in xrange(NUM_OF_IMAGES) :
		str_label = f.readline().strip()
		str_labels.append(str_label)
		data = np.zeros((IMAGE_H, IMAGE_W, 1), dtype = "float32")
		for r in xrange(IMAGE_H) :
			cols = f.readline().split(",")
			for j in xrange(IMAGE_W) :
				data[r, j, 0] = np.float32(cols[j])

		datas.append(data)

		labels_0.append(getLabelFromStr(str_label[0]))
		labels_1.append(getLabelFromStr(str_label[1]))
		labels_2.append(getLabelFromStr(str_label[2]))
		labels_3.append(getLabelFromStr(str_label[3]))


train_datas = np.stack(datas[0:len(datas) * 9 / 10])
print train_datas.shape
train_labels0 = np.stack(labels_0[0:len(labels_0) * 9 / 10])
train_labels1 = np.stack(labels_1[0:len(labels_1) * 9 / 10])
train_labels2 = np.stack(labels_2[0:len(labels_2) * 9 / 10])
train_labels3 = np.stack(labels_3[0:len(labels_3) * 9 / 10])
print train_labels0.shape
print train_labels1.shape
print train_labels2.shape
print train_labels3.shape


test_datas = np.stack(datas[len(datas) * 9 / 10 : len(datas)])
test_str_labels = str_labels[len(str_labels) * 9 / 10 : len(str_labels)]


inputs = Input(shape=(35, 80, 1))
model = Sequential()
model.add(Convolution2D(10, 3, 3, border_mode='valid', input_shape=(35, 80, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
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

# x = merge([x1, x2, x3, x4], mode = "concat")

# output = Dense(40, activation='softmax')(x)

model = Model(input=inputs, output=[x1, x2, x3, x4])

from keras.utils.visualize_util import plot
plot(model, show_shapes=True, to_file='simple_cnn_multi_output_model.png')

model.compile(loss='categorical_crossentropy',
			  loss_weights=[1., 1., 1., 1.],
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(train_datas, [train_labels0, train_labels1, train_labels2, train_labels3], nb_epoch=50, batch_size=32)

outputs = model.predict(test_datas)

def get_code_from_output (output) :
	ret = ""
	for i in xrange(0, 4):
		maxV = 0
		maxI = 0
		for j in xrange(0, 9):
			if output[i][j] >= maxV:
				maxV = output[i][j]
				maxI = j
		ret += str(maxI)
	return ret

print len(outputs)

correct = 0
for i in xrange(len(outputs[0])) :
	# print test_str_labels[i] + "," + get_code_from_output(outputs[i])
	x = test_str_labels[i]
	y = get_code_from_output([outputs[0][i], outputs[1][i], outputs[2][i], outputs[3][i]])
	print x, y
	if x == y:
		correct = correct + 1

# total: 500, correct:108
print "total: " + str(len(outputs[0])) + ", correct:" + str(correct)





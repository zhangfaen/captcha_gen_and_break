from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Activation, Dropout
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D


import numpy as np

datas = []
labels = []
str_labels = []

with open("/Users/baidu/dev/go_workspace/src/github.com/zhangfaen/captcha/capgen/xx/data.csv") as f:
	for line in f:
		cols = line.split(",")
		cols_len = len(cols)
		data = np.array(cols[1:], dtype = "float32")
		datas.append(data)
		str_labels.append(cols[0])
		label = np.zeros(4 * 10, dtype = "float32")
		index = 0
		for c in cols[0]:
			for i in xrange(0, 10) :
				if c != str(i):
					label[index] = 0
				else:
					label[index] = 1.0 / 4
				index = index + 1
		# print cols[0], "  ", label
		labels.append(label)


train_datas = np.vstack(datas[0:len(datas) * 9 / 10])
train_labels = np.vstack(labels[0:len(labels) * 9 / 10])

test_datas = np.vstack(datas[len(datas) * 9 / 10 : len(datas)])
test_labels = np.vstack(labels[len(labels) * 9 / 10 : len(labels)])
test_str_labels = str_labels[len(str_labels) * 9 / 10 : len(str_labels)]

model = Sequential()
model.add(Dense(100, input_dim=35 * 80, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(40, activation='softmax'))


from keras.utils.visualize_util import plot
plot(model, show_shapes=True, to_file='model.png')

def my_accuracy(y_true, y_pred):
    #print (y_true)
    #print (y_pred)
    return {
        'false_neg': 1,
        'false_pos': 2,
    }


model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(np.vstack(train_datas), np.vstack(train_labels), nb_epoch=2000, batch_size=32)

outputs = model.predict(test_datas)

def get_code_from_output (output) :
	ret = ""
	for i in xrange(0, 4):
		maxV = 0
		maxI = 0
		for j in xrange(0, 9):
			if output[i* 10 + j] >= maxV:
				maxV = output[i*10 + j]
				maxI = j
		ret += str(maxI)
	return ret

print len(outputs)

correct = 0
for i in xrange(len(outputs)) :
	# print test_str_labels[i] + "," + get_code_from_output(outputs[i])
	if test_str_labels[i] == get_code_from_output(outputs[i]):
		correct = correct + 1

print "total: " + str(len(outputs)) + ", correct:" + str(correct)




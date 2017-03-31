# captcha_gen_and_break
利用这个Github repo，讲一讲如何用卷积神经网络，识别如下的验证码图片。下面是两个要识别的验证码的例子
![zhangfaen](https://github.com/zhangfaen/captcha_gen_and_break/blob/master/gen/4_test/100.png) ![zhangfaen](https://github.com/zhangfaen/captcha_gen_and_break/blob/master/gen/4_test/101.png)

针对上述两张图片，我们要训练的深度学习模型能够识别他们为两个字符串 "6017"和"2013" 

用于训练的数据集通常很宝贵，我们这个题目相对来说，容易获得训练集。我在repo里专门写了一个脚本，用来生成训练集，可以执行如下命令
    $python gen/gen.py 默认会产生5000张4个字符的验证码图片。你也可以手工改一下gen.py脚本，来产生更多图片或者更多字符的验证码图片（本来应该提供参数控制，但我偷懒了~~）。
    
    
    def get_cnn_net(num, _shape):
    """
        Define the CNN network model.
        This model has 3 cov lays and 4 outputs.
        Each output is for one digit in the image.
        Finally we optimize the 4 output at the same time, each ouput contribute 1/4 the total loss.
    """

    inputs = Input(shape=_shape)
    conv1 = Convolution2D(32, 5, 5, border_mode='same',
                          activation='relu')(inputs)
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
    model.compile(loss='categorical_crossentropy', loss_weights=[1.]
                  * num, optimizer='Adam', metrics=['accuracy'])
    return model

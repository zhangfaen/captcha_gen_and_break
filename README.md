# captcha_gen_and_break
利用这个https://github.com/zhangfaen/captcha_gen_and_break/，
讲一讲如何用卷积神经网络，识别如下的验证码图片。下面是两个要识别的验证码的例子


![zhangfaen](https://github.com/zhangfaen/captcha_gen_and_break/blob/master/gen/4_test/100.png) ![zhangfaen](https://github.com/zhangfaen/captcha_gen_and_break/blob/master/gen/4_test/101.png)

针对上述两张图片，我们要训练的深度学习模型能够识别他们为两个字符串 "6017"和"2013" 

用于训练的数据集通常很宝贵，我们这个题目相对来说，容易获得训练集。我在repo里专门写了一个脚本，用来生成训练集，可以执行如下命令
    $python gen/gen.py 默认会产生5000张4个字符的验证码图片。你也可以手工改一下gen.py脚本，来产生更多图片或者更多字符的验证码图片（本来应该提供参数控制，但我偷懒了~~）。


接下来我们使用Tensorflow上的Keras库定义一个卷积神经网络模型，代码如下。
    
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
以上代码定义了一个卷积神经网络模型，简单解释一下。它有3个卷积层，每层都是32个kernels；第一层卷积之前，加了padding。这个模型是多输出的，总共有4个输出，4个输出同时优化，每个输出的权重占1/4.


模型图如下 
![image](https://github.com/zhangfaen/captcha_gen_and_break/blob/master/break/cnn.png)


定义好模型结构后，我们导入数据，开始训练和预测。经过20次迭代后，5万张图片的训练集，预测精度大概能到95%。

# 放个广告 ：）
一开始我用的笔记本训练，i7的CPU，大概需要2个小时的训练时间。百度云的深度学习平台是我做的 https://cloud.baidu.com/product/bdl.html 。里面可以创建GPU虚机集群，我用了一个GPU卡的虚机，再训练了一次这个模型，用时大概在3分钟左右，加速比大概40倍。那个GPU卡的虚机启动后，自带Tensorflow和Jupiter，非常方便使用，每个小时5元。哈哈


未来，百度云深度学习平台还会提供其他GPU套餐，比如一个虚机多个GPU卡，和多个虚机的集群。GPU卡也会有更多种，比如NV的P40. 这个深度学习平台为了还会提供集群资源监控、管理系统；也会提供作业提交系统，这样，你提交一个作业后，就可以去休息一下，百度云深度学习平台会自动申请资源，训练任务，把结果存到BOS，然后释放集群。

# 按需使用，杜绝浪费！

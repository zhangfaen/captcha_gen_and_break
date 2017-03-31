# captcha_gen_and_break
利用这个Github repo，讲一讲如何用卷积神经网络，识别如下的验证码图片。下面是两个要识别的验证码的例子
![zhangfaen](https://github.com/zhangfaen/captcha_gen_and_break/blob/master/gen/4_test/100.png) ![zhangfaen](https://github.com/zhangfaen/captcha_gen_and_break/blob/master/gen/4_test/101.png)

针对上述两张图片，我们要训练的深度学习模型能够识别他们为两个字符串 "6017"和"2013" 

用于训练的数据集通常很宝贵，我们这个题目相对来说，容易获得训练集。我在repo里专门写了一个脚本，用来生成训练集，可以执行如下命令
    $python gen/gen.py 默认会产生5000张4个字符的验证码图片。你也可以手工改一下gen.py脚本，来产生更多图片或者更多字符的验证码图片（本来应该提供参数控制，但我偷懒了~~）。
    
    

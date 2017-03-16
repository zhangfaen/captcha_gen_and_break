import random
import sys
import os
from image import ImageCaptcha


def gen_4(size,output):
    '''
    generate captcha pngs with four numbers
    '''
    image = ImageCaptcha(width=160, height=60, fonts=['comic.ttf'], font_sizes=(46, 58, 68))
    if not os.path.exists(output):
        os.mkdir(output)
    labels_path = output + '/' + 'labels.txt'
    labels_file = open(labels_path, 'w')
    for i in xrange(size):
        label = str(random.randint(1000, 9999))
        image.write(label, output + '/' + str(i) + '.png')
        labels_file.write(label + '\n')
    labels_file.close()

def gen_6(size,output):
    '''
    generate captcha pngs with six numbers
    '''
    image = ImageCaptcha(width=180, height=60, fonts=['comic.ttf'], font_sizes=(38, 46, 49))
    if not os.path.exists(output):
        os.mkdir(output)
    labels_path = output + '/' + 'labels.txt'
    labels_file = open(labels_path, 'w')
    for i in xrange(size):
        label = str(random.randint(100000, 999999))
        image.write(label, output + '/' + str(i) + '.png')
        labels_file.write(label + '\n')
    labels_file.close()

if __name__ == '__main__':
    size = 5000
    train_ratio = 0.9
    num_figure = '4'
    output_prefix = '4'
    #num = sys.argv[1]
    #num_figure = sys.argv[2]
    #output_prefix = sys.argv[3] 
    if(num_figure == '4'):
        output_train = output_prefix + '_train'
        output_test = output_prefix + '_test'
        train_size  = int(size * train_ratio)
        test_size  = size - train_size
        gen_4(train_size,output_train)
        gen_4(test_size,output_test)
    elif (num_figure == '6') :
        output_train = output_prefix + '_train'
        output_test = output_prefix + '_test'
        train_size  = int(size * train_ratio)
        test_size  = size - train_size
        gen_6(train_size,output_train)
        gen_6(test_size,output_test)
    else :
        print 'error'
        sys.exit(-1)

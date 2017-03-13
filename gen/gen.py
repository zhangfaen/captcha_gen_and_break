import random

from image import ImageCaptcha

# png image counts
num = 20

# generate captcha pngs with four numbers
image = ImageCaptcha(width=160, height=60, fonts=['comic.ttf'], font_sizes=(46, 58, 68))
data_dir = 'captcha_data4/'
labels_path = data_dir + 'labels.txt'
labels_file = open(labels_path, 'w')
for i in xrange(num):
    label = str(random.randint(1000, 9999))
    image.write(label, data_dir + str(i) + '.png')
    labels_file.write(label + '\n')
labels_file.close()

'''
# generate captcha pngs with six numbers
image = ImageCaptcha(width=180, height=60, fonts=['comic.ttf'], font_sizes=(38, 46, 49))
data_dir = 'captcha_data6/'
labels_path = data_dir + 'labels.txt'
labels_file = open(labels_path, 'w')
for i in xrange(num):
    label = str(random.randint(100000, 999999))
    image.write(label, data_dir + str(i) + '.png')
    labels_file.write(label + '\n')
labels_file.close()
'''

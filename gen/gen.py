from image import ImageCaptcha
import random
image = ImageCaptcha(fonts=['comic.ttf'])
data_dir = 'pydata/'
labels_path = data_dir + 'labels.txt'
labels_file = open(labels_path,'w')
for i in xrange(5000) :
    label=str(random.randint(1000,9999))
    image.write(label, data_dir + str(i) + '.png')
    labels_file.write(label + '\n')
labels_file.close()
    
    
    

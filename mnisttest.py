import os
import sys
import numpy as np
import matplotlib as plt
import caffe

MODEL_FILE = '/home/hades/machinelearning/caffe-master/examples/mnist/lenet.prototxt'
PRETRAINED = '/home/hades/machinelearning/caffe-master/examples/mnist/lenet_iter_10000.caffemodel'
IMAGE_FILE = '/home/hades/machinelearning/caffe-master/examples/mnist/test_im.bmp'

imput_image = caffe.io.load_image(IMAGE_FILE,color = False);
print imput_image
net = caffe.Classifier(MODEL_FILE,PRETRAINED);
prediction = net.predict([imput_image],oversample = False);
caffe.set_mode_gpu();
print 'predict class:',prediction[0].argmax()

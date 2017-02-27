#! usr/bin/env python

#this a pythonprogram that conver mnist bainry data into image

import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image

filename = '/home/hades/machinelearning/caffe-master/data/mnist/t10k-images-idx3-ubyte' 
binfile = open(filename,'rb');
buf = binfile.read();

index = 0;
magic, numImages, numRows, numColumes = struct.unpack_from('>IIII',buf,index);
index += struct.calcsize('>IIII');

for image in range(0,numImages):
	im = struct.unpack_from('>784B',buf,index);
	index += struct.calcsize('>784B');
	im = np.array(im,dtype='uint8');
	im = im.reshape(28,28);
	im = Image.fromarray(im)
	im.save('mnist_test_train_%s.bmp'%image,'bmp')


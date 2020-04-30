import sys
import caffe
import numpy as np
import cv2

WIDTH = 160
HEIGHT = 160

#load architecture and parameters
net = caffe.Net('inception_resnet_v1_test.prototxt', 'inception_resnet_v1.caffemodel', caffe.TEST)

# load input and configure preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

#transposes image coming from opencv
#opencv: [height (0), width (1), channels (2)]
#caffe: [batch size (x), channels (2), height (0), width (1)]
transformer.set_transpose('data', (2,0,1))

#add batch size (1, since we test just one image)
net.blobs['data'].reshape(1,3,HEIGHT,WIDTH)

#load image
img_name = 'rahul.jpg'
img = cv2.imread(img_name)
img = cv2.resize(img,(HEIGHT,WIDTH))

#load the image in the data layer
net.blobs['data'].data[...] = transformer.preprocess('data', img)

#compute forward pass
output = net.forward()

print('output:')
print(len(output['InceptionResnetV1/Bottleneck/BatchNorm/FusedBatchNorm'][0]))

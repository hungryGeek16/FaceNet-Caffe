import caffe
import sys
import numpy as np


# Load weights
a = np.load('weights.npy',allow_pickle=True)
# Convert it into dictionary
data = a.tolist()
# Extract layer names
layers = list(data.keys())


#Initialize prototxt file
net = caffe.Net('inception_resnet_v1_test.prototxt',caffe.TEST)

# To Read all layers respective weights.
for i in layers: 
    print(i)
    if i == "InceptionResnetV1/Bottleneck/MatMul":
        net.params[i][0].data[...] = data[i]['weights'].reshape(512,1792) 
    elif(i.find('FusedBatchNorm') == -1):
        net.params[i][0].data[...] = data[i]['weights'].transpose((3,2,0,1)) 
    else:
        net.params[i][0].data[...] = data[i]['scale']
        net.params[i][1].data[...] = data[i]['bias']




# Save caffemodel.
net.save('inception_resnet_v1.caffemodel')

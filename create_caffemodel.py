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

bias_a = 1
bias_b = 1
bias_c = 1
# To Read all layers respective weights.
for i in layers: 
    print(i)
    if i.count('weights'):
        net.params[i.strip('/weights')+'/Conv2d'][0].data[...] = data[i].transpose((3,2,0,1))
    if i.count('BatchNorm'):
        if i.count('beta'):
            net.params[i.strip('/beta')+'/Scale'][0].data[...] = data[i]
	
        if i.count('Const'):
            net.params[i.strip('/Const')+'/Bias'][0].data[...] = data[i] 
        
        if i.count('moving_mean'):
            net.params[i.strip('/moving_mean')+'/FusedBatchNorm'][0].data[...] = data[i]
        if i.count('moving_variance'):
            net.params[i.strip('/moving_variance')+'/FusedBatchNorm'][0].data[...] = data[i]
    if i.count('biases'):
        if i.count('block35'):
            net.params['BiasA'+str(bias_a)][0].data[...] = data[i]  
            bias_a+=1
        if i.count('block17'):
            net.params['BiasB'+str(bias_b)][0].data[...] = data[i]  
            bias_b+=1
        if i.count('block8'):
            net.params['BiasC'+str(bias_c)][0].data[...] = data[i]  
            bias_c+=1                		
# Save caffemodel.
net.save('inception.caffemodel')

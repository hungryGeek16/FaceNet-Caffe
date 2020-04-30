# Process Followed to Convert Tensorflow model to Caffe Model:

1. Define prototxt file for caffe model, layer names defined in prototxt file must be same as of defined in tensorflow model.```This might become easy while loading weights into deep nets.``` 

2. Extract weights in the form of .npy format.```Used mmconvert to extract all layer's weights```. [Link](https://github.com/microsoft/MMdnn) to mmconvert.

3. Now load params defined for each and every layer into the layers defined in prototxt file by importing weights.npy file into the program.
## Format of .npy file
**Conv Layer**

```python
{'weights': array([[[[ 5.98237850e-02, -2.07922816e-01, -1.39808946e-03,
           -3.19408774e-02, -8.60046297e-02,  7.20547587e-02,
            5.69821596e-02, -1.02225527e-01,  2.84959804e-02,
            1.94275483e-01,  3.35525163e-02, -6.59388304e-02,
            ................................................]]]])}
of shape (num_of_filters,height,width,channels)
```
**Batch Normalization**

```python
{'scale': array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ..........................................................], dtype=float32),
 'bias': array([-0.42156017, -0.3743814 , -0.41813663, -0.2842422 , -0.22323455,
        -0.4140735 , -0.32575113, -0.36014694, -0.3688915 , -0.25125268,
        ................................................], dtype=float32),
 'mean': array([ 0.02131204,  0.01151724,  0.07367857,  0.05918641,  0.03779788,
         0.03275746,  0.0228732 ,  0.03388122, -0.00890283,  0.0702192 ,
         0.01396199,  0.05063147,  0.02793424,  0.07309864, -0.02325859,
        -0.01633305,  0.04464114,  0.0549158 ,  0.05630469,  0.03060376,
         0.10651793,  0.06386697,  0.00867011,  0.04050072,  0.05344747,
         ..............................................], dtype=float32),
 'var': array([0.00759798, 0.00898295, 0.00776405, 0.00819074, 0.01009827,
        0.00966888, 0.00870785, 0.00734035, 0.00886426, 0.00940575,
        ...................................................................)}
 Each normalization parameters are equal to the number of filters defined for every convolution.
 The bias,mean and var are trained parameters which are meant to be used during inferencing.
```

4. Load each and every parameter as given below
```python
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
````
5. After saving caffemodel, it's ready for inferencing.The model outputs 512 nodes, in which it uses 128 for embeddings and rest of them are used for the predicted label.

# Challenges faced During Conversion

a.
1. No proper documentation found on the internet to extract weights from frozen.pb model.
2. Tried to load that model using ckpt and index files, but for some reasons the files are shown as invalid.
3. Because of these errors, used mmconvert but never reads bias layer weights.

<p align="center">
<img src = "/Net/ss.png">
</p>

b. Batch Normalization does not accept mean and var parameters as whole bunch, there's a shape error.  

<p align="center">
<img src = "/Net/ss1.png">
</p>  

# Proposed Solution:
b.  **A bias and a scale layer should be initialized for every batch normalization layer.**

### How to run
1. Download and keep all files in the same folder
2. The type these following commands to create caffemodel and start inferencing:
```python
python create_caffemodel.py # For creation of caffemodel
python caffe_test.py # For inferencing
```

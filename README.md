# SRCNN_keras


A implementation of the original paper ['Image Super-Resolution Using Deep Convolutional Networks'](https://arxiv.org/abs/1501.00092)



<center><img width = "800" src="https://user-images.githubusercontent.com/58276840/94503875-77722b00-0242-11eb-85f8-93e7cb0fdd11.png"></center>





### Implementations
-------------------------------------------------------
My implementation may have some differences with the original paper:


#### Networks)

Patch extraction and prepresentation)
- Filter size: 9x9
- Channel maps: 64
- Activation function: ReLU
- Border mode: Same (Zero Padding)

Non-linear mapping)
- Filter size: 5x5
- Channel mapls: 32
- Activation function: ReLU
- Border mode: Same (Zero Padding)

Patch extraction and prepresentation)
- Filter size : 5x5
- Channel maps: 1
- Activation function: Linear
- Border mode: Same (Zero Padding)


#### Training)

- Loss Function: MSE (Mean Squared Error)
- Optimizer: Adam
- Learning rate: 0.0001 (0.00001 for the last layer)
- Batch size: 128


#### Dataset)

##### Training)
- 91 images (Training set) Set5 (Validation set)
- Patch size: 33 (Input, label)

##### Test)
- Set5, Set14, BSD100, Urban100


### Use
-------------------------------------------------------

You can generate dataset (training sample, test sample) through matlab files in Dataset_Generating directory
- Excute for training sample: 'SRCNN_generate_train.m'
- Excute for test sample: 'generate_test.m'


I also uploaded the trained weight files.

With SRCNN.ipynb files and weight files, you can test the network.
- Excute for x2 SRCNN: SRCNN_scale2.ipynb, /weights/SRCNN_checkpoint_scale2.h5
- Excute for x3 SRCNN: SRCNN_scale3.ipynb, /weights/SRCNN_checkpoint_scale3.h5
- Excute for x4 SRCNN: SRCNN_scale4.ipynb, /weights/SRCNN_checkpoint_scale4.h5

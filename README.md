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


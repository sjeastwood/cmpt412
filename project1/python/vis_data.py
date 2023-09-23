import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
import os


# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)
m_train = xtrain.shape[1]

batch_size = 1
layers[0]['batch_size'] = batch_size

img = xtest[:,0]
img = np.reshape(img, (28, 28), order='F')
plt.imshow(img.T, cmap='gray')
plt.show()
plt.close()

output = convnet_forward(params, layers, xtest[:,0:1])
output_1 = np.reshape(output[0]['data'], (28,28), order='F')

##### Fill in your code here to plot the features ######
resultsdir = '../results'
os.makedirs(resultsdir, exist_ok=True)

fig, ax = plt.subplots(1,1)
ax.imshow(output_1.T, cmap='gray')
ax.set_axis_off()

fig.suptitle('Original Image')
filename = f"{resultsdir}/original_image.png"
plt.savefig(filename)

fig, ax = plt.subplots(4,5)

data_size = output[1]['height'] * output[1]['width']
start = 0
end = 0
#4x5 sub plot
for row in range(4):
    for col in range(5):

        end = start + data_size

        img_data = output[1]['data'][start:end]
        img = np.reshape(img_data, (output[1]['height'],output[1]['width']), order='F')
        
        ax[row,col].imshow(img.T, cmap='gray') 
        ax[row,col].set_axis_off()

        start = end

        

fig.suptitle("Outputs from Convolution Layer - Layer 2")
filename = f"{resultsdir}/second_layer_outputs.png"
plt.savefig(filename)

fig, ax = plt.subplots(4,5)

data_size = output[2]['height'] * output[2]['width']
start = 0
end = 0
#4x5 sub plot
for row in range(4):
    for col in range(5):

        end = start + data_size

        img_data = output[2]['data'][start:end]
        img = np.reshape(img_data, (output[2]['height'],output[2]['width']), order='F')
        
        ax[row,col].imshow(img.T, cmap='gray') 
        ax[row,col].set_axis_off()

        start = end

        

fig.suptitle("Outputs from ReLu Layer - Layer 3")
filename = f"{resultsdir}/third_layer_outputs.png"
plt.savefig(filename)
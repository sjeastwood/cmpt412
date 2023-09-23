import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


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

custom_image_folder = "../results/"

raw_img = []

raw_img.append(Image.open(custom_image_folder + 'drawn1.png'))
raw_img.append(Image.open(custom_image_folder + 'drawn2.png'))
raw_img.append(Image.open(custom_image_folder + 'drawn3.png'))
raw_img.append(Image.open(custom_image_folder + 'drawn4.png'))
raw_img.append(Image.open(custom_image_folder + 'drawn4_2.png'))
raw_img.append(Image.open(custom_image_folder + 'drawn7.png'))
raw_img.append(Image.open(custom_image_folder + 'drawn7_2.png'))
raw_img.append(Image.open(custom_image_folder + 'drawn9.png'))

invert_img = []
for i in range(len(raw_img)):
    gr_img = ImageOps.grayscale(raw_img[i])
    invert_img.append(ImageOps.invert(gr_img)) 
# colours needed to be inverted as this network appears to be trained 
# where white represents presence of a character and black is 0

for i in range(len(invert_img)):
    invert_img[i] = invert_img[i].resize((28,28))

img_arr = []
for i in range(len(invert_img)):
    img_arr.append((np.asarray(invert_img[i]))/255) #normalize between 0 and 1

batch_size = 1
layers[0]['batch_size'] = batch_size

my_pred = []
for i in range(len(img_arr)):

    cptest, P = convnet_forward(params, layers, img_arr[i], test=True)
    my_pred.append(P)

my_pred_val = []
# go through all the y_preds and get the index representing the predicted digit and store it
for i in range(len(my_pred)):
    my_pred_val.append(np.argmax(my_pred[i]))

print(my_pred_val)
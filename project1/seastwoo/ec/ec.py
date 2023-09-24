import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import sys
import cv2

# To grab the other python files required
sys.path.insert(1, '../python/')

from utils import get_lenet
from load_mnist import load_mnist
from conv_net import convnet_forward
from init_convnet import init_convnet

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

#Load data
image_dir = "../images/"

images = []
image_name = []

file_list = os.listdir(image_dir)

for file in file_list:
    if "jpg" in file or "JPG" in file or "png" in file or "PNG" in file:
        img = cv2.imread(image_dir+file, cv2.IMREAD_GRAYSCALE)
        images.append(img)
        image_name.append(file)


# Convert the images to binary with a threshold

images_binary = []

for image in images:
    # Blur the image
    blur = cv2.GaussianBlur(image,(7,7),0)
    #grab each image based on threshold
    t_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21 , 25)
    images_binary.append(t_img)


# Truth Values
img_truth = []
img_truth.append("[2, 0, 6, 6, 6]")
img_truth.append("[9, 6, 7, 3, 8, 0, 5, 4, 2, 1]")
img_truth.append("[5, 9, 6, 8, 7, 0, 3, 4, 2, 1]")
img_truth.append("[2, 1, 0, 1, 4, 5, 4, 9, 7, 9, 6, 0, 1, 5, 3, 0, 9, 4, 9, 7, 6, 6, 1, 5, 0, 4, 0, 9, 4, 7, 1, 3, 2, 1, 2, 1, 3, 4, 7, 7, 2, 2, 3, 1, 1, 5, 4, 7, 4, 4]")


# Go through each image, extract the digits and then feed it through the network

batch_size = 1
layers[0]['batch_size'] = batch_size

for i in range(len(images)):
    digit_imgs = [] #stores the digits this image composes of

    #grab the image name for print out
    img_name = image_name[i]

    #grab the binary image
    img = images_binary[i]

    #Find the connected components of the image

    num, _, box_stats, cpts = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)

    for j in range(1,num):

        left = box_stats[j, cv2.CC_STAT_LEFT]
        right = left + box_stats[j, cv2.CC_STAT_WIDTH]-5
        top = box_stats[j, cv2.CC_STAT_TOP]
        bottom = top + box_stats[j, cv2.CC_STAT_HEIGHT]-10
        extract_digit = img[top:bottom,left:right]

        #pad the image
        pad_digit = cv2.copyMakeBorder(extract_digit, 9, 9, 9, 9, cv2.BORDER_CONSTANT, value=0)
        #resize image
        size_digit = cv2.resize(pad_digit,(28,28), interpolation=cv2.INTER_AREA)

        #add the image to array for queuing into the model
        digit_imgs.append(size_digit)
        # print("size of image {}".format(size_digit.shape))
    
    print(img_name)
    # print("number of digits found in image {}".format(len(digit_imgs)))

    #Feed the images into the network

    my_pred = []

    for k in range(len(digit_imgs)):
        cptest, P = convnet_forward(params, layers, digit_imgs[k], test=True)
        my_pred.append(P)

    my_pred_val = []
    # go through all the y_preds and get the index representing the predicted digit and store it
    for k in range(len(my_pred)):
        my_pred_val.append(np.argmax(my_pred[k]))
    
    print("Truth Values: ")
    print(img_truth[i])

    print("Image Predictions array: ")
    print(my_pred_val)
    print("---------------------------------")



#Referenced: https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
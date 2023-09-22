import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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


# Testing the network
#### Modify the code to get the confusion matrix ####
all_preds = []
for i in range(0, xtest.shape[1], 100):
    cptest, P = convnet_forward(params, layers, xtest[:,i:i+100], test=True)
    all_preds.append(P)

# hint: 
#     you can use confusion_matrix from sklearn.metrics (pip install -U scikit-learn)
#     to compute the confusion matrix. Or you can write your own code :)

# print(confusion_matrix(ytest, P))


############ FROM chatGPT ################
# from sklearn.metrics import confusion_matrix

# # Initialize an empty list to store all the predictions and true labels
# all_preds = []
# true_labels = []

# # Loop through the test data in batches of 100 as you did before
# for i in range(0, xtest.shape[1], 100):
#     cptest, P = convnet_forward(params, layers, xtest[:, i:i+100], test=True)
    
#     # Append the batch predictions to the list of predictions
#     all_preds.append(P)
    
#     # Append the true labels of this batch to the list of true labels
#     true_labels.extend(ytest[:, i:i+100])

# # Combine all the predictions into a single numpy array
# y_pred = np.concatenate(all_preds, axis=1)

# # Flatten the list of true labels into a 1D numpy array
# y_true = np.array(true_labels).flatten()

# # Calculate the confusion matrix
# confusion = confusion_matrix(y_true, np.argmax(y_pred, axis=0))

# # Print or use the confusion matrix as needed
# print(confusion)

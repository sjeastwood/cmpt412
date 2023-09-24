import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
ground_truth = []
for i in range(0, xtest.shape[1], 100):
    cptest, P = convnet_forward(params, layers, xtest[:,i:i+100], test=True)
    all_preds.append(P)
    ground_truth.extend(ytest[:, i:i+100])
    
# Combine all the predictions into a single numpy array
y_pred = np.concatenate(all_preds, axis=1)

y_pred_val = []

# now go through all the y_preds and get the index representing the predicted digit and store it
for i in range(y_pred.shape[1]):
    y_pred_val.append(np.argmax(y_pred[:,i]))

y_pred_flatten = np.array(y_pred_val).flatten() #make it a 1d array

y_true = np.array(ground_truth).flatten() #make it a 1d array

labels=[0,1,2,3,4,5,6,7,8,9]

cm = confusion_matrix(y_true, y_pred_flatten, labels=labels)

cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
cm_disp.plot()

#convert to matplotlib figure and save it to results
cm_disp.figure_.suptitle('Confusion Matrix')

cm_disp.figure_.savefig('../results/confusion_matrix.png',dpi=300)



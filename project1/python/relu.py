import numpy as np

def activate(x):
    # if x > 0:
    #     return x
    # else:
    #     return 0
    return np.maximum(0,x)
    
def differentiate(x):
    if x >= 0: #greater than 0 vs non-negative
        return 1
    else:
        return 0

def relu_forward(input_data):
    output = {
        'height': input_data['height'],
        'width': input_data['width'],
        'channel': input_data['channel'],
        'batch_size': input_data['batch_size'],
    }

    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    # output['data'] = np.zeros_like(input_data['data'])

    # vectorize the function to perform relu across all elements in the input array
    relu_activate = np.vectorize(activate)
    output['data'] = relu_activate(input_data['data']) #np.maximum(0,input_data['data'])
    
    return output

def relu_backward(output, input_data, layer):
    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    # input_od = np.zeros_like(input_data['data'])

    # print(input_data['data'].shape)

    # relu_diff = np.vectorize(differentiate)
    dR = np.where(input_data['data'] >= 0, 1, 0)#relu_diff(input_data['data'])
    input_od = np.multiply(output['diff'],dR)

    # print(input_od.shape)

    return input_od

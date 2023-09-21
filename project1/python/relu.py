import numpy as np

def activate(x):
    if x > 0:
        return x
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
    output['data'] = relu_activate(input_data['data'])
    
    return output

def relu_backward(output, input_data, layer):
    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    input_od = np.zeros_like(input_data['data'])

    return input_od

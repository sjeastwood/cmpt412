import numpy as np


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    d, k = input["data"].shape
    n = param["w"].shape[1]

    ###### Fill in the code here ######
    W = param["w"]
    X = input["data"]
    b = param["b"]
    
    f = np.dot(W.T,X) + b.T
    
    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": f #np.zeros((n, k)) # replace 'data' value with your implementation
    }

    return output


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.

    input = input_data['data']

    param_grad['b'] = output['diff'] #since partial derivate with respect to b is 1
    param_grad['w'] = np.dot(input, output['diff'].T)
    input_od = np.dot(param['w'], output['diff'])

    return param_grad, input_od
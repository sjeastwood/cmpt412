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
    
    # print("x shape {}".format(X.shape))
    # print("W shape {}".format(W.shape))
    # print("b shape {}".format(b.shape))
    f = np.dot(W.T,X) + b.T

    # print(f.shape)
    
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

    batch_size = output['batch_size']

    db = np.zeros((1,batch_size)) #make a batch_size x 1 vector to multiply with output['diff']

    param_grad['b'] = np.matmul(db, output['diff'].T) 
    param_grad['w'] = np.matmul(input, output['diff'].T)
    input_od = np.matmul(param['w'], output['diff'])

    return param_grad, input_od
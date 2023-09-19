import numpy as np
from utils import im2col_conv_batch

def pooling_layer_forward(input, layer):
    """
    Forward pass for the pooling layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.
    """
    
    h_in = input['height']
    w_in = input['width']
    c = input['channel']
    batch_size = input['batch_size']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']

    h_out = int((h_in + 2 * pad - k) / stride + 1)
    w_out = int((w_in + 2 * pad - k) / stride + 1)
    
    ###### Fill in the code here ######
    output_data = np.zeros((c*h_out*w_out,batch_size))

    for batch in range(batch_size):
        #input image reshaped
        img = input_data['data'][:,batch].reshape(c, h_in, w_in) #need to account for padding

        # add padding if it exists
        pad_img = np.zeros((c,pad*2+h_in,pad*2+w_in))
        for chan in range(c):
            pad_img[chan] = np.pad(img[chan],pad)

        #matrix to hold the pooled layers
        pool_mtx = np.zeros((c,h_out,w_out))

        #slide kernel across pad_img and grab max

        row = 0
        col = 0 #to control the pool_mtx row and col to be filled in
        #go across rows with j
        for j in range(0,pad_img.shape[1],stride): #stride is 2

            col = 0 #reset column for next row

            #go across columns with i
            for i in range(0,pad_img.shape[2],stride):

                #go across channels
                for chan in range(c):
                    #grab max from the kernel slice of image mtx
                    pool_mtx[chan,row,col] = np.max(pad_img[chan,j:j+stride,i:i+stride])

                col = col + 1 # move on to the next column

            row = row + 1 # increment row when columns are done

        output_data[:,batch] = pool_mtx.reshape(1,c*h_out*w_out)[0]
    
    output = {}
    output['height'] = h_out
    output['width'] = w_out
    output['channel'] = c
    output['batch_size'] = batch_size
    output['data'] = output_data#np.zeros((h_out, w_out, c, batch_size)) # replace with your implementation

    return output

def pooling_layer_backward(output, input, layer):
    """
    Backward pass for the pooling layer.

    Parameters:
    - output (dict): Contains the gradients from the next layer.
    - input (dict): Contains the original input data.
    - layer (dict): Layer configuration containing parameters such as kernel size, padding, stride, etc.

    Returns:
    - input_od (numpy.ndarray): Gradient with respect to the input.
    """

    h_in = input['height']
    w_in = input['width']
    c = input['channel']
    batch_size = input['batch_size']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']

    h_out = (h_in + 2*pad - k) // stride + 1
    w_out = (w_in + 2*pad - k) // stride + 1

    input_od = np.zeros(input['data'].shape)
    input_od = input_od.reshape(h_in * w_in * c * batch_size, 1)

    im_b = np.reshape(input['data'], (h_in, w_in, c, batch_size), order='F')
    im_b = np.pad(im_b, ((pad, pad), (pad, pad), (0, 0), (0, 0)), mode='constant')
    
    diff = np.reshape(output['diff'], (h_out*w_out, c*batch_size), order='F')

    for h in range(h_out):
        for w in range(w_out):
            matrix_hw = im_b[h*stride : h*stride + k, w*stride : w*stride + k, :, :]
            flat_matrix = matrix_hw.reshape((k*k, c*batch_size), order='F')
            i1 = np.argmax(flat_matrix, axis=0)
            R, C = np.unravel_index(i1, matrix_hw.shape[:2], order='F')
            nR = h*stride + R
            nC = w*stride + C
            i2 = np.ravel_multi_index((nR, nC), (h_in, w_in), order='F')
            i4 = np.ravel_multi_index((i2, np.arange(c*batch_size)), (h_in*w_in, c*batch_size), order='F')
            i3 = np.ravel_multi_index((h, w), (h_out, w_out), order='F')
            input_od[i4] += diff[i3:i3+1, :].T

    input_od = np.reshape(input_od, (h_in*w_in, c*batch_size), order='F')
    input_od = np.reshape(input_od, (h_in*w_in*c, batch_size), order='F')

    return input_od

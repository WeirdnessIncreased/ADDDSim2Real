import numpy as np

def numpy_conv(inputs, filter, _result, padding="VALID"):
    H, W = inputs.shape
    filter_size = filter.shape[0]
    # default np.floor
    filter_center = int(filter_size / 2.0)
    filter_center_ceil = int(np.ceil(filter_size / 2.0))

    result = np.zeros((_result.shape))
    H, W = inputs.shape
    #print("new size",H,W)
    for r in range(0, H - filter_size + 1):
        for c in range(0, W - filter_size + 1):
            cur_input = inputs[r : r + filter_size, c : c + filter_size]
            cur_output = cur_input * filter
            conv_sum = np.sum(cur_output)
            result[r, c] = conv_sum
    return result
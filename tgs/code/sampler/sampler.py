import numpy as np
import util_image 

def sample(inputs):
    if len(inputs) == 2:
        return sample_pair(inputs)
    elif len(inputs) == 1:
        return sample_x(inputs[0])
    else:
        pass

def hwc2chw(data):
    return data.transpose((2,0,1)).astype(np.float32)

def sample_x(x):
    x = np.repeat(x, 3, axis=2)
    x = hwc2chw(x)
    return x

def sample_t(t):
    t = hwc2chw(t)
    return t

def sample_pair(inputs):
    x, t = inputs
    x = util_image.resize(x,(128,128,3),mode='reflect', preserve_range=True)
    t = util_image.resize(t,(128,128,1),mode='reflect', preserve_range=True)

    x = hwc2chw(x)
    t = hwc2chw(t)
    return x, t


sample_train = sample
sample_val = sample

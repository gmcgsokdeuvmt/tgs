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
    x = util_image.resize(x,(202,202,3),mode='reflect', preserve_range=True)
    x = util_image.do_center_pad_to_factor(x, factor=32)
    x = hwc2chw(x)
    return x

def sample_t(t):
    t = util_image.resize(t,(202,202,1),mode='reflect', preserve_range=True)
    t = util_image.do_center_pad_to_factor(t, factor=32)
    t = hwc2chw(t)
    return t

def sample_pair(inputs):
    x, t = inputs
    x = util_image.resize(x,(202,202,3),mode='reflect', preserve_range=True)
    t = util_image.resize(t,(202,202,1),mode='reflect', preserve_range=True)
    x, t = util_image.do_center_pad_to_factor2(x, t, factor=32)

    x = hwc2chw(x)
    t = hwc2chw(t)
    return x, t


sample_train = sample
sample_val = sample

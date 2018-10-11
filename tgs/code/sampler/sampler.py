import numpy as np
import util_image 

def sample(inputs):
    if len(inputs) == 2:
        return sample_pair(inputs)
    else:
        return sample_x(inputs)

def hwc2chw(data):
    return data.transpose((2,0,1)).astype(np.float32)

def hwc2chw_pair(inputs):
    x, t = inputs
    x = x.transpose((2,0,1)).astype(np.float32)
    t = t.transpose((2,0,1)).astype(np.float32)
    return x, t

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

def resize202_pair(inputs):
    x, t = inputs
    x = util_image.resize(x,(202,202,1),mode='reflect', preserve_range=True)
    t = util_image.resize(t,(202,202,1),mode='reflect', preserve_range=True)
    return x, t

def pad256_pair(inputs):
    x, t = inputs
    return util_image.do_center_pad_to_factor2(x, t, factor=64)

def repeatx3(x):
    return np.concatenate([x.copy(),x.copy(),x.copy()],axis=2)

def pre_sample(inputs):
    if len(inputs) == 2:
        return pre_sample_pair(inputs)
    else:
        return pre_sample_x(inputs)

def pre_sample_pair(inputs):
    inputs = resize202_pair(inputs)
    return inputs

def aug_sample_pair(inputs):
    return inputs

def post_sample_pair(inputs,aug=False):
    if aug:
        inputs = aug_sample_pair(inputs)
    inputs = pad256_pair(inputs)
    x, t = inputs
    x = repeatx3(x)
    inputs = (x, t)
    inputs = hwc2chw_pair(inputs)
    return inputs
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

def add_depth_channels(x):
    h, w, _ = x.shape
    for row, const in enumerate(np.linspace(0, 1, h)):
        x[row, :, 1] = const
    x[:,:,2] = x[:,:,0] * x[:,:,1]
    return x

def pre_sample(inputs):
    if len(inputs) == 2:
        return pre_sample_pair(inputs)
    else:
        return pre_sample_x(inputs)

def pre_sample_pair(inputs):
    inputs = resize202_pair(inputs)
    return inputs

def post_sample_pair(inputs,aug=False):
    if aug:
        inputs = train_augment(inputs)
    inputs = pad256_pair(inputs)
    x, t = inputs
    x = repeatx3(x)
    x = add_depth_channels(x)
    inputs = (x, t)
    inputs = hwc2chw_pair(inputs)
    return inputs

def train_augment(inputs):
    image, mask = inputs
    if np.random.rand() < 0.5:
        image, mask = util_image.do_horizontal_flip2(image, mask)

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image, mask = util_image.do_random_shift_scale_crop_pad2(image, mask, 0.125)
        if c==1:
            image, mask = util_image.do_elastic_transform2(image, mask, grid=10,
                                            distort=np.random.uniform(0,0.1))
        if c==2:
            image, mask = util_image.do_shift_scale_rotate2( image, mask, dx=0, dy=0, scale=1,
                                                angle=np.random.uniform(0,10))

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image = util_image.do_brightness_shift(image,np.random.uniform(-0.05,+0.05))
        if c==1:
            image = util_image.do_brightness_multiply(image,np.random.uniform(1-0.05,1+0.05))
        if c==2:
            image = util_image.do_gamma(image,np.random.uniform(1-0.05,1+0.05))

    return image,mask
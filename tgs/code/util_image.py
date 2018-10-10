import numpy as np
import os

from skimage.io import imread
from skimage import img_as_uint, io

from skimage.transform import resize

import cv2

# もっといい書き方？
def search_images(dir,suffix='.png'):
    return [ 
        (img_name[:-len(suffix)] , os.path.join(dir,img_name)) 
        for img_name in os.listdir(dir) 
        if img_name.endswith(suffix) ]

def load_images(dir,suffix='.png',img_dtype=np.uint8,return_ids=False,offset=None,size=None):
    image_paths = None
    if (offset is None) or (size is None):
        image_paths = search_images(dir,suffix)
    else:
        image_paths = search_images(dir,suffix)[offset:(offset+size)]
    images = [ imread(fname=img_path,as_gray=True) for _, img_path in image_paths ]
    if img_dtype == np.uint8:
        images = np.asarray(images,dtype=np.float32)/255.
        images = images[:,:,:,:1]
    elif img_dtype == np.uint16:
        images = np.asarray(images,dtype=np.float32)/65535.
        images = images[:,:,:,np.newaxis]
    else:
        images = np.asarray(images,dtype=np.float32)
    
    if return_ids:
        ids = [ img_id for img_id, _ in image_paths ]
        return ids, images
    else:
        return images

def resize_images(images, shape=(128,128,1)):
    resized_images = [ resize(img, shape, mode='reflect', preserve_range=True) for img in images ]
    return resized_images

def gen_depth_kernels(mu=[0.5],sigma=0.1):
    depth_weights = []
    for i in np.arange(0,1,0.1):
        c1 = stats.norm.cdf(i,loc=mu,scale=sigma)
        c2 = stats.norm.cdf(i+0.1,loc=mu,scale=sigma)
        depth_weights.append(c2-c1)
    
    depth_weights = np.asarray(depth_weights)
    depth_weights = depth_weights/depth_weights.sum(axis=0)[np.newaxis,:]
    return depth_weights

def normalize_images(images):
    normalized_images = [ (img - img.mean())/img.std() if img.std() > 1e-5 else (img-img.mean()) for img in images]
    return normalized_images

def compute_center_pad(H,W, factor=32):

    if H%factor==0:
        dy0,dy1=0,0
    else:
        dy  = factor - H%factor
        dy0 = dy//2
        dy1 = dy - dy0

    if W%factor==0:
        dx0,dx1=0,0
    else:
        dx  = factor - W%factor
        dx0 = dx//2
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1

def do_center_pad_to_factor(image, factor=32):
    H,W = image.shape[:2]
    dy0, dy1, dx0, dx1 = compute_center_pad(H,W, factor)

    image = cv2.copyMakeBorder(image, dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
                            #cv2.BORDER_CONSTANT, 0)
    return image[:,:,np.newaxis]

def do_repad_randomly2(image, mask, H, W, factor=32): # (c,h,w)

    # transpose -> hwc
    image = image.transpose((1,2,0))
    mask = mask.transpose((1,2,0))

    dy0, dy1, dx0, dx1 = compute_center_pad(H,W,factor=factor)
    orig_img = image[dy0:(dy0+H),dx0:(dx0+W),:].copy()
    orig_mask = mask[dy0:(dy0+H),dx0:(dx0+W),:].copy()

    long_H = dy0+dy1+H
    long_W = dx0+dx1+W

    # shift and random pad
    dy0 = np.random.randint(dy0+dy1)
    dy1 = long_H - dy0 - H
    dx0 = np.random.randint(dx0+dx1)
    dx1 = long_W - dx0 - W

    image = cv2.copyMakeBorder(orig_img, dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(orig_mask, dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)  

    # transform -> chw
    image = image[np.newaxis,:,:]
    mask = mask[np.newaxis,:,:]
    return image, mask               


def do_center_pad_to_factor2(image, mask, factor=32):
    image = do_center_pad_to_factor(image, factor)
    mask  = do_center_pad_to_factor(mask, factor)
    return image, mask

def do_center_pad_images(images, factor=32):
    padded_images = [ do_center_pad_to_factor(img,factor=factor) for img in images ]
    return padded_images

def do_crop_center_image(image, offH, offW): # chw
    return image[:,offH:-offH,offW:-offW].copy()
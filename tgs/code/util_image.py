import numpy as np
import os

from skimage.io import imread
from skimage import img_as_uint, io

from skimage.transform import resize

import cv2
import random
import math

def do_resize2(image, mask, H, W):
    image = cv2.resize(image,dsize=(W,H))
    mask = cv2.resize(mask,dsize=(W,H))
    mask  = (mask>0.5).astype(np.float32)

    return image,mask

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
        if len(images.shape) == 3:
            images = images[:,:,:,np.newaxis]
        else:
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
    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]
    return image

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


#---

def do_horizontal_flip(image):
    #flip left-right
    image = cv2.flip(image,1)
    return image

def do_horizontal_flip2(image,mask):
    image = do_horizontal_flip(image)
    mask  = do_horizontal_flip(mask )
    return image, mask

#---

def compute_random_pad(H,W, limit=(-4,4), factor=32):


    if H%factor==0:
        dy0,dy1=0,0
    else:
        dy  = factor - H%factor
        dy0 = dy//2 + np.random.randint(limit[0],limit[1]) # np.random.choice(dy)
        dy1 = dy - dy0

    if W%factor==0:
        dx0,dx1=0,0
    else:
        dx  = factor - W%factor
        dx0 = dx//2 + np.random.randint(limit[0],limit[1]) # np.random.choice(dx)
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1


def do_random_pad_to_factor2(image, mask, limit=(-4,4), factor=32):
    H,W = image.shape[:2]
    dy0, dy1, dx0, dx1 = compute_random_pad(H,W, limit, factor)

    image = cv2.copyMakeBorder(image, dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
    mask  = cv2.copyMakeBorder(mask,  dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)

    return image, mask

#----
def do_invert_intensity(image):
    #flip left-right
    image = np.clip(1-image,0,1)
    return image


def do_brightness_shift(image, alpha=0.125):
    image = image + alpha
    image = np.clip(image, 0, 1)
    return image


def do_brightness_multiply(image, alpha=1):
    image = alpha*image
    image = np.clip(image, 0, 1)
    return image


#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):

    image = image ** (1.0 / gamma)
    image = np.clip(image, 0, 1)
    return image


def do_flip_transpose2(image, mask, type=0):
    #choose one of the 8 cases

    if type==1: #rotate90
        image = image.transpose(1,0)
        image = cv2.flip(image,1)

        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,1)


    if type==2: #rotate180
        image = cv2.flip(image,-1)
        mask  = cv2.flip(mask,-1)


    if type==3: #rotate270
        image = image.transpose(1,0)
        image = cv2.flip(image,0)

        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,0)


    if type==4: #flip left-right
        image = cv2.flip(image,1)
        mask  = cv2.flip(mask,1)


    if type==5: #flip up-down
        image = cv2.flip(image,0)
        mask  = cv2.flip(mask,0)

    if type==6:
        image = cv2.flip(image,1)
        image = image.transpose(1,0)
        image = cv2.flip(image,1)

        mask = cv2.flip(mask,1)
        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,1)

    if type==7:
        image = cv2.flip(image,0)
        image = image.transpose(1,0)
        image = cv2.flip(image,1)

        mask = cv2.flip(mask,0)
        mask = mask.transpose(1,0)
        mask = cv2.flip(mask,1)


    return image, mask

##================================
def do_shift_scale_crop( image, mask, x0=0, y0=0, x1=1, y1=1 ):
    #cv2.BORDER_REFLECT_101
    #cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    image = image[y0:y1,x0:x1]
    mask  = mask [y0:y1,x0:x1]

    image = cv2.resize(image,dsize=(width,height))
    mask  = cv2.resize(mask,dsize=(width,height))
    mask  = (mask>0.5).astype(np.float32)
    return image, mask


def do_random_shift_scale_crop_pad2(image, mask, limit=0.10):

    H, W = image.shape[:2]

    dy = int(H*limit)
    y0 =   np.random.randint(0,dy)
    y1 = H-np.random.randint(0,dy)

    dx = int(W*limit)
    x0 =   np.random.randint(0,dx)
    x1 = W-np.random.randint(0,dx)

    #y0, y1, x0, x1
    image, mask = do_shift_scale_crop( image, mask, x0, y0, x1, y1 )
    return image, mask

#===========================================================================

def do_shift_scale_rotate2( image, mask, dx=0, dy=0, scale=1, angle=0 ):
    borderMode=cv2.BORDER_REFLECT_101
    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    sx = scale
    sy = scale
    cc = math.cos(angle/180*math.pi)*(sx)
    ss = math.sin(angle/180*math.pi)*(sy)
    rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

    box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ],np.float32)
    box1 = box0 - np.array([width/2,height/2])
    box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat  = cv2.getPerspectiveTransform(box0,box1)

    image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = cv2.warpPerspective(mask, mat, (width,height),flags=cv2.INTER_NEAREST,#cv2.INTER_LINEAR
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask  = (mask>0.5).astype(np.float32)
    return image, mask

#https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations
#https://github.com/letmaik/lensfunpy/blob/master/lensfunpy/util.py
def do_elastic_transform2(image, mask, grid=32, distort=0.2):
    borderMode=cv2.BORDER_REFLECT_101
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width,np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end   = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step*(1+random.uniform(-distort,distort))

        xx[start:end] = np.linspace(prev,cur,end-start)
        prev=cur


    y_step = int(grid)
    yy = np.zeros(height,np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end   = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step*(1+random.uniform(-distort,distort))

        yy[start:end] = np.linspace(prev,cur,end-start)
        prev=cur

    #grid
    map_x,map_y =  np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    #image = map_coordinates(image, coords, order=1, mode='reflect').reshape(shape)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=borderMode,borderValue=(0,0,0,))


    mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=borderMode,borderValue=(0,0,0,))
    mask  = (mask>0.5).astype(np.float32)
    return image, mask




def do_horizontal_shear2( image, mask, dx=0 ):
    borderMode=cv2.BORDER_REFLECT_101
    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    dx = int(dx*width)

    box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ],np.float32)
    box1 = np.array([ [+dx,0], [width+dx,0],  [width-dx,height], [-dx,height], ],np.float32)

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0,box1)

    image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask  = cv2.warpPerspective(mask, mat, (width,height),flags=cv2.INTER_NEAREST,#cv2.INTER_LINEAR
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask  = (mask>0.5).astype(np.float32)
    return image, mask
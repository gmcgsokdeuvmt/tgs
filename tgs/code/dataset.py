import numpy as np
import os

import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

import time
import util_image
from sampler import sampler
from joblib import Parallel, delayed

class Dataset(torch.utils.data.Dataset):

    def __init__(self,images=None,masks=None):
        self.images = images
        self.masks = masks
        self.image_ids = None
        self.mask_ids = None
        self.suffix = '.png'
        self.pre_sample_func  = sampler.pre_sample_pair
        self.post_sample_func = sampler.post_sample_pair

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        record = [
            e[idx]
            for e in [self.images, self.masks] if e is not None
        ]

        return record

    def load_images_and_masks(self,image_path,mask_path):
        self.load_images(image_path)
        self.load_masks(mask_path)
        return self

    def load_images(self,dir,offset=None,size=None):
        image_ids, images = util_image.load_images(dir,suffix=self.suffix,img_dtype=np.uint8,return_ids=True,offset=offset,size=size)

        self.images = images
        self.image_ids = image_ids
        return self
    
    def load_masks(self,dir):
        image_ids, images = util_image.load_images(dir,suffix=self.suffix,img_dtype=np.uint16,return_ids=True)

        self.masks = images
        self.mask_ids = image_ids
        return self

    def get_loader(self,batch_size=10,shuffle=False,drop_last=False):
        return torch.utils.data.DataLoader( dataset=self,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            drop_last=drop_last
                                            )

    def split(self,val_percentage=0.1,stratify=None,shuffle=False):
        if stratify is not None:
            shuffle = True

        image_ids_train, image_ids_test, mask_ids_train, mask_ids_test, images_train, images_test, masks_train, masks_test = train_test_split(
            self.image_ids,
            self.mask_ids,
            self.images,
            self.masks,
            test_size=val_percentage, stratify=stratify, shuffle=shuffle)
        
        train_set = Dataset(images=images_train,masks=masks_train)
        train_set.image_ids = image_ids_train
        train_set.mask_ids = mask_ids_train
        train_set.pre_sample_func  = sampler.pre_sample_pair
        train_set.post_sample_func = sampler.post_sample_pair

        val_set = Dataset(images= images_test,masks=masks_test)
        val_set.image_ids = image_ids_test
        val_set.mask_ids = mask_ids_test
        val_set.pre_sample_func  = sampler.pre_sample_pair
        val_set.post_sample_func = sampler.post_sample_pair

        return train_set, val_set

    def select_ids(self,ids):
        new_images = []
        new_masks = []
        for id in ids:
            pos = self.image_ids.index(id)
            new_images.append(self.images[pos])
            new_masks.append(self.masks[pos])
        
        new_set = Dataset(images=new_images,masks=new_masks)
        new_set.image_ids = ids
        new_set.mask_ids = ids
        new_set.pre_sample_func  = self.pre_sample_func
        new_set.post_sample_func = self.post_sample_func
        return new_set
    
    def presample_pair(self):
        new_images = []
        new_masks  = []
        for idx in range(len(self.images)):
            record = self[idx]
            
            image, mask = self.pre_sample_func(record)
            
            new_images.append(image)
            new_masks.append(mask)
        
        new_set = Dataset(images=new_images,masks=new_masks)
        new_set.image_ids = self.image_ids
        new_set.mask_ids = self.mask_ids
        new_set.pre_sample_func  = self.pre_sample_func
        new_set.post_sample_func = self.post_sample_func
        return new_set

    def presample_pair_parallel(self):
        new_images = []
        new_masks  = []
        new_records = Parallel(n_jobs= -1)\
            (delayed(self.pre_sample_func)(self[idx]) for idx in range(len(self.images)))

        for idx in range(len(self.images)):
            image, mask = new_records[idx]
            new_images.append(image)
            new_masks.append(mask)
        
        new_set = Dataset(images=new_images,masks=new_masks)
        new_set.image_ids = self.image_ids
        new_set.mask_ids = self.mask_ids
        new_set.pre_sample_func  = self.pre_sample_func
        new_set.post_sample_func = self.post_sample_func
        return new_set

    def presample_image(self):
        new_images = []
        for idx in range(len(self.images)):
            record = self[idx]
            
            image = self.pre_sample_func(record)
            
            new_images.append(image)
        
        new_set = Dataset(images=new_images)
        new_set.image_ids = self.image_ids
        new_set.pre_sample_func  = self.pre_sample_func
        new_set.post_sample_func = self.post_sample_func
        return new_set
    
    def presample_image_parallel(self):
        new_images = []
        new_records = Parallel(n_jobs= -1)\
            (delayed(self.pre_sample_func)(self[idx]) for idx in range(len(self.images)))

        for idx in range(len(self.images)):
            image = new_records[idx]
            new_images.append(image)
        
        new_set = Dataset(images=new_images)
        new_set.image_ids = self.image_ids
        new_set.pre_sample_func  = self.pre_sample_func
        new_set.post_sample_func = self.post_sample_func
        return new_set

    def postsample_pair_parallel(self,aug=False):
        new_images = []
        new_masks  = []
        new_records = Parallel(n_jobs= -1)\
            (delayed(self.post_sample_func)(self[idx],aug=aug) for idx in range(len(self.images)))

        for idx in range(len(self.images)):
            image, mask = new_records[idx]
            new_images.append(image)
            new_masks.append(mask)
        
        new_set = Dataset(images=new_images,masks=new_masks)
        new_set.image_ids = self.image_ids
        new_set.mask_ids = self.mask_ids
        new_set.pre_sample_func  = self.pre_sample_func
        new_set.post_sample_func = self.post_sample_func
        return new_set
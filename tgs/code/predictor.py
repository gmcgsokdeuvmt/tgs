import torch
import numpy as np
from torch.autograd import Variable
import dataset
from skimage import img_as_uint, io
import os
import util_image
import cv2

def calc_iou(y_true, y_hat):
    intersect = np.logical_and(y_true, y_hat).sum()
    union = np.logical_or(y_true, y_hat).sum()
    if union > 0:
        return intersect / union
    else:
        return 1

class Predictor:
    def __init__(self,model):
        self.model = model
        self.temp_path = 'pred_temp'

    def ___predict_core(self, images, predict_half=False):
        if predict_half:
            # up left 
            images00 = images[:,:,:64,:64]
            images00 = Variable(images00.cuda())
            batch_preds00 = torch.sigmoid(self.model(images00))

            # up right
            images01 = images[:,:,:64,64:128]
            images01 = Variable(images01.cuda())
            batch_preds01 = torch.sigmoid(self.model(images01))

            # down left
            images10 = images[:,:,64:128,:64]
            images10 = Variable(images10.cuda())
            batch_preds10 = torch.sigmoid(self.model(images10))

            # down right
            images11 = images[:,:,64:128,64:128]
            images11 = Variable(images11.cuda())
            batch_preds11 = torch.sigmoid(self.model(images11))

            # concat
            batch_preds = torch.cat([
                torch.cat([batch_preds00, batch_preds01], dim=3),
                torch.cat([batch_preds10, batch_preds11], dim=3)
            ],dim=2)
        
        else:
            images = Variable(images.cuda())
            batch_preds = torch.sigmoid(self.model(images))

        return batch_preds

    def __predict_core(self, dataset, batch_size=40, predict_half=False,mode=None):
        data_loader = dataset.get_loader(batch_size=batch_size,shuffle=False)

        has_mask = dataset.masks is not None
        if has_mask:
            self.true_masks = dataset.masks
            if mode=='pad':
                H, W = 202, 202
                dy0, dy1, dx0, dx1 = util_image.compute_center_pad(H,W)
                self.true_masks = [mask[:,dy0:(dy0+H),dx0:(dx0+W)].copy() for mask in self.true_masks]
                self.true_masks = [cv2.resize(mask[0,:,:], dsize=(101,101))[np.newaxis,:,:]  for mask in self.true_masks]
                # self.true_masks = [util_image.resize(mask, (1,101,101), mode='reflect', preserve_range=True)  for mask in self.true_masks]
                

        preds = []
        for images in data_loader:
            images = images[0]

            batch_preds = self.___predict_core(images, predict_half=predict_half)

            for i, _ in enumerate(images):
                pred = batch_preds[i]
                pred = pred.cpu().data.numpy()
                if mode=='pad':
                    H, W = 202, 202
                    dy0, dy1, dx0, dx1 = util_image.compute_center_pad(H,W)
                    pred = pred[:,dy0:(dy0+H),dx0:(dx0+W)].copy()
                    pred = cv2.resize(pred[0,:,:], dsize=(101,101))[np.newaxis,:,:]
                    #pred = util_image.resize(pred, (1,101,101), mode='reflect', preserve_range=True)

                preds.append(pred)
        

        return preds

    def predict(self, dataset, batch_size=40, predict_half=False,mode=None,flip=False):
        self.preds = self.__predict_core(dataset,batch_size=batch_size, predict_half=predict_half,mode=mode)
        if flip:
            self.preds = np.asarray(self.preds)[:,:,:,::-1]
        return self

    def predict_file(self, dataset, batch_size=40, save_dir=None, predict_half=False,mode=None,flip=False):
        if save_dir is None:
            save_dir = self.temp_path

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.preds = self.__predict_core(dataset,batch_size=batch_size, predict_half=predict_half,mode=mode)

        for i in range(len(self.preds)):
                id = dataset.image_ids[i]
                pred = self.preds[i]
                pred = pred[0,:,:]*255
                pred = pred.astype(np.uint8)
                if flip:
                    pred = pred[:,::-1].copy()
                io.imsave('./{}/{}.png'.format(save_dir,id), img_as_uint(pred))
                
        return self

    def mask(self, threshold):

        masks = [ (pred > threshold)[0,:,:] for pred in self.preds ]
        self.masks = masks
        return self

    def eval_ious(self, thresholds=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]):
        
        ious = []
        for y_hat, y_true in zip(self.masks, self.true_masks):
            iou = calc_iou(y_hat, y_true)
            ious.append(iou)
        
        ious = np.asarray(ious)

        value = np.mean([ ious > t for t in thresholds ]) # if match all then 1.
        self.value = value
        return self

    def resize(self,shape=(101,101)):
        self.preds = dataset.util_image.resize_images(self.preds,shape=shape)
        return self
            

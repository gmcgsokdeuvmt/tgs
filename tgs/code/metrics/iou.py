import torch.nn.functional as F
import numpy as np

def calc_iou(y_hat, y_true):
    intersect = np.logical_and(y_true, y_hat).sum()
    union = np.logical_or(y_true, y_hat).sum()
    if union > 0:
        return intersect / union
    else:
        return 1

class IOUEvaluator():
        def __init__(self, thresholds=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],reduction='elementwise_mean',pre_sigmoid=True):
            self.thresholds = thresholds
            self.reduction = reduction
            self.pre_sigmoid = pre_sigmoid

        def __call__(self, y_preds, y_trues):
            if self.pre_sigmoid:
                y_preds = F.sigmoid(y_preds)

            ious = []
            for y_pred, y_true in zip(y_preds.cpu().data.numpy(), y_trues.cpu().data.numpy()):
                iou = calc_iou(y_pred>0.5, y_true>0.5)
                ious.append(iou)
            
            ious = np.asarray(ious)
            ious = np.asarray([ ious > t for t in self.thresholds ])

            if self.reduction == 'elementwise_mean':
                value = np.mean(ious) # if match all then 1.
                return value
            else:
                value = np.mean(ious,axis=0)
                return value
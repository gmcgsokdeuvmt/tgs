from losses import focal_loss
from metrics import iou

config_focal = {
    'focal_loss': focal_loss.FocalLoss2d(size_average=True),
    'accuracy'       : iou.IOUEvaluator(pre_sigmoid=True, reduction='elementwise_mean')
}

config_focal_coef = {
    'focal_loss': 1
}

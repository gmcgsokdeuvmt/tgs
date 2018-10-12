from losses import focal_loss, lovasz_losses
from metrics import iou

config_focal = {
    'focal_loss': focal_loss.FocalLoss2d(size_average=True),
    'accuracy'       : iou.IOUEvaluator(pre_sigmoid=True, reduction='elementwise_mean')
}

config_focal_coef = {
    'focal_loss': 1
}

# - - -

config_lovasz = {
    'lovasz_loss': lovasz_losses.lovasz_hinge,
    'accuracy'       : iou.IOUEvaluator(pre_sigmoid=True, reduction='elementwise_mean')
}

config_lovasz_coef = {
    'lovasz_loss': 1
}

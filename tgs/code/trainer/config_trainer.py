from evaluator import evaluator_focal, evaluator_lovasz

evaluators = {
    'focal': evaluator_focal,
    'lovasz': evaluator_lovasz
}

config_log = [
    'epoch',
    'train_loss',
    'train_acc',
    'val_loss',
    'val_acc'
]
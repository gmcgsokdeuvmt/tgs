from evaluator import evaluator_focal, evaluator_lovasz

evaluator = evaluator_focal
#evaluator = evaluator_lovasz

config_log = [
    'epoch',
    'train_loss',
    'train_acc',
    'val_loss',
    'val_acc'
]
from torch.optim.lr_scheduler import *

class EarlyStopping():
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self,
                 min_delta=0,
                 patience=5):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs
        Arguments
        ---------
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvment before terminating.
            the counter be reset after each improvment
        """
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e15
        self.stopped_epoch = 0

    def step(self, loss):
        current_loss = loss
        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    return True
                self.wait += 1
        return False

class CheckBestModel():
    def __init__(self):
        self.best_metric = -1e15

    def step(self, metric):
        current_metric = metric
        if current_metric is None:
            pass
        else:
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                return True
                
        return False

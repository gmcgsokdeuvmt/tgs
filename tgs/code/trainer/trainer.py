import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from skimage import filters
import time

from trainer import config_trainer
evaluator = config_trainer.evaluator

from sampler import sampler
class Trainer:
    def __init__(self,train_dataset,val_dataset,
                optimizer):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer   = optimizer

    def epoch_train(self,model,train_loader):
        losses = []
        accs = []

        for i, batch in enumerate(train_loader):
            input_x, input_t = batch

            input_x = Variable(input_x.cuda())
            input_t = Variable(input_t.cuda())

            eval_result = evaluator.evaluate(model, input_x, input_t)
            total_loss  = evaluator.calc_total_loss(eval_result)
            losses.append(total_loss.data.clone())
            accs.append(eval_result['accuracy'])
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        loss =  np.mean(losses)
        acc = np.mean(accs)
        return loss, acc

    def epoch_val(self,model,val_loader):
        losses = []
        accs = []

        for i, batch in enumerate(val_loader):
            input_x, input_t = batch

            input_x = Variable(input_x.cuda())
            input_t = Variable(input_t.cuda())

            eval_result = evaluator.evaluate(model, input_x, input_t)
            total_loss  = evaluator.calc_total_loss(eval_result)
            losses.append(total_loss.data.clone())
            accs.append(eval_result['accuracy'])

        loss =  np.mean(losses)
        acc = np.mean(accs)
        return loss, acc

    def train(self,model,epoch_num,batch_size=16):
        model.cuda()
        for epoch in range(epoch_num):
            t = time.time()

            aug_train_set = self.train_dataset.postsample_pair_parallel(aug=True)

            train_loader = aug_train_set.get_loader(batch_size=batch_size,shuffle=True,drop_last=True)
            val_loader   = self.val_dataset.get_loader(batch_size=batch_size,shuffle=False,drop_last=False)
            
            train_loss, train_acc = self.epoch_train(model, train_loader)
            val_loss, val_acc = self.epoch_val(model, val_loader)

            print('Epoch {} is done! ({}s)'.format(epoch, time.time()-t))
            print('Epoch: {}. \t Train Loss: {:.4g}. \t Val Loss: {:.4g}'.format(epoch, train_loss, val_loss))
            print('Epoch: {}. \t Train Acc : {:.4g}. \t Val Acc : {:.4g}'.format(epoch, train_acc, val_acc))

        
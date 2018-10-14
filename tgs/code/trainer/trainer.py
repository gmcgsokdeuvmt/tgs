import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from skimage import filters
import time

from trainer import config_trainer
evaluators = config_trainer.evaluators

from sampler import sampler
import gc
import codecs
import util_scheduler

class Trainer:
    def __init__(self,train_dataset,val_dataset,
                optimizer,eval_type='focal'):
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.optimizer     = optimizer
        self.checkpoints   = 'ch{}.pth'
        self.best_checkpoints   = 'best_ch{}.pth'
        self.log_filename  = 'train.log'
        self.evaluator     = evaluators[eval_type]

    def epoch_train(self,model,train_loader,actual_batch_rate=1):
        losses = []
        accs = []
    
        for i, batch in enumerate(train_loader):
            input_x, input_t = batch

            input_x = Variable(input_x.cuda())
            input_t = Variable(input_t.cuda())

            eval_result = self.evaluator.evaluate(model, input_x, input_t)
            total_loss  = self.evaluator.calc_total_loss(eval_result)
            losses.append(total_loss.data.clone())
            accs.append(eval_result['accuracy'])
            
            if (i % actual_batch_rate) == 0:
                self.optimizer.zero_grad()
            (total_loss/actual_batch_rate).backward()
            if (i % actual_batch_rate) == (actual_batch_rate - 1):
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

            eval_result = self.evaluator.evaluate(model, input_x, input_t)
            total_loss  = self.evaluator.calc_total_loss(eval_result)
            losses.append(total_loss.data.clone())
            accs.append(eval_result['accuracy'])

        loss =  np.mean(losses)
        acc = np.mean(accs)
        return loss, acc

    def train(self,model,epoch_num,batch_size=16,actual_batch_rate=1,is_scheduler=False,cyclic_scheduler=False):
        model.cuda()
        checkBestModel    = util_scheduler.CheckBestModel()
        earlyStopping     = util_scheduler.EarlyStopping(patience=16)
        reduceLROnPlateau = util_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=8)
        cosineAnnealingLR = util_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=0.001)
        for epoch in range(epoch_num):
            t = time.time()

            aug_train_set = self.train_dataset.postsample_pair_parallel(aug=True)

            train_loader = aug_train_set.get_loader(batch_size=batch_size,shuffle=True,drop_last=True)
            val_loader   = self.val_dataset.get_loader(batch_size=batch_size,shuffle=False,drop_last=False)
            
            model.train()
            train_loss, train_acc = self.epoch_train(model, train_loader, actual_batch_rate)
            model.eval()
            val_loss, val_acc = self.epoch_val(model, val_loader)

            lr = None
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = float(param_group['lr'])
                break

            print('Epoch {} is done! ({}s)'.format(epoch, time.time()-t))
            print('Epoch: {}. \t Train Loss: {:.4g}. \t Val Loss: {:.4g}'.format(epoch, train_loss, val_loss))
            print('Epoch: {}. \t Train Acc : {:.4g}. \t Val Acc : {:.4g}'.format(epoch, train_acc, val_acc))
            gc.collect()
            #save_path = self.checkpoints.format(epoch)
            #torch.save(model.state_dict(), save_path)
            #print('  Save model: {}'.format(save_path))

            print(
                epoch,
                '{:.4g}'.format(lr),
                '{:.4g}'.format(train_loss),
                '{:.4g}'.format(val_loss),                
                '{:.4g}'.format(train_acc),
                '{:.4g}'.format(val_acc),
                sep=",", 
                end="\n", 
                file=codecs.open(self.log_filename, 'a+', 'utf-8'), 
                flush=True
            )

            
            save_best_path = self.best_checkpoints.format(epoch)
            if checkBestModel.step(val_acc):
                torch.save(model.state_dict(), save_best_path)
                print('  Save Best model: {}'.format(save_best_path))

            if is_scheduler:
                if earlyStopping.step(val_loss):
                    print('  Early Stopping!')
                    break
                reduceLROnPlateau.step(val_loss)

            if cyclic_scheduler:
                cosineAnnealingLR.step()


    
            
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from .abs_trainer import Trainer


class Syn1dTrainer(Trainer):

    def __init__(self, model, train_loader, valid_loader, test_loader, config):
        self.global_step = 0
        self.epoch = 0
        #self.max_step = config.max_epoch * config.step_per_epoch
        #self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        self.max_epoch = config.train.max_epoch
        self.min_lr = config.train.min_lr
        super().__init__(model, train_loader, valid_loader, test_loader, config)
    
    def get_optimizer(self):
        print(self.config.optimizer)
        if self.config.optimizer.type == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.optimizer.weight_decay,
                                          betas=(self.config.optimizer.beta1, self.config.optimizer.beta2))
            return optimizer
        else:
            raise NotImplementedError
    
    def get_scheduler(self, optimizer):
        print(self.config.scheduler)
        if self.config.scheduler.type == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    "min",
                    factor=self.config.scheduler.factor,
                    patience=self.config.scheduler.patience,
                    min_lr=self.min_lr,
                )  
            return {
                    'scheduler': lr_scheduler,
                    'frequency': 'epoch'
                }   
        else:
            raise NotImplementedError

    def lr_weight(self, step):
        if self.global_step >= self.config.warmup:
            return 0.99 ** self.epoch
        return (self.global_step + 1) * 1.0 / self.config.warmup
    
    def train_step(self, batch):
        return self.step(batch, type='train')
    
    def val_step(self, batch):
        return self.step(batch, type='val')
    
    def test_step(self, batch):
        return self.step(batch, type='test')
    
    def step(self, batch, type='train'):
        labels = batch['y'].squeeze().long()
        logits = self.model(batch['smiles'].long())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
 
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().sum()
        self.log(f'Loss/{type}', loss)
        self.log(f'Acc/{type}', accuracy)
        if type == 'train':
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr)
        #print(predictions.sum())
        return loss, accuracy

    

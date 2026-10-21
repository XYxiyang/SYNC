import torch
import torch.nn as nn
import torch.nn.functional as F
from .abs_trainer import Trainer


class Syn3dRebuildTrainer(Trainer):

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
                    min_lr=self.min_lr
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
        labels = batch.y.squeeze().long()
        matrice = self.model(batch)
        criterion_matrice = nn.CrossEntropyLoss()
        loss_matrice = criterion_matrice(matrice.view(-1, 6), batch.adj_matrice.flatten().long()) / len(batch)
        acc = ((torch.argmax(matrice, dim=-1) == batch.adj_matrice).sum() - batch.adj_mask.sum()) / (len(batch.adj_mask.view(-1)) - batch.adj_mask.sum())
        #acc = (torch.argmax(matrice, dim=-1) == batch.adj_matrice).sum() / len(batch)

        #self.log(f'Loss_Logits/{type}', loss_logits)
        self.log(f'Loss_Matrice/{type}', loss_matrice)
        #self.log(f'Acc/{type}', accuracy)
        if type == 'train':
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr)
        #print(predictions.sum())
        #print(f"acc: {accuracy}, loss_logits: {loss_logits}, loss_matrice: {loss_matrice}")

        return  loss_matrice, acc

    
 
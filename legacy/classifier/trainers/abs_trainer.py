import os
import re
import json
from tqdm import tqdm
import gc
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time 
from utils.logger import print_log
from synsbdd.data.collate import Batch
import time

  
class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, config):
        self.model = model
        #self.model_ema = ModelEma(model, decay=config.ema_decay)
        self.config = config.train
        self.optimizer = self.get_optimizer()
        sched_config = self.get_scheduler(self.optimizer)
        if sched_config is None:
            sched_config = {
                'scheduler': None,
                'frequency': None
            }
        self.scheduler = sched_config['scheduler']
        self.sched_freq = sched_config['frequency']
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # distributed training
        self.local_rank = -1

        # log
        self.tag = self.config.tag if self.config.tag != '' else 'version'
        self.version = self._get_version()

        self.config.save_dir = os.path.join(self.config.save_dir, f'{self.tag}_{self.version}')
        self.model_dir = os.path.join(self.config.save_dir, 'checkpoint')
        # initialize right before training
        self.writer_buffer = {}
        self.writer = SummaryWriter(self.config.save_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        with open(os.path.join(self.config.save_dir, 'config.json'), 'w') as fout:
            json.dump(config.__dict__, fout)

        # training process recording
        self.valid_requires_grad = False
        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.topk_ckpt_file = os.path.join(self.model_dir, 'topk_map.txt')
        self.topk_ckpt_map = []    # smaller index means better ckpt
        #self.patience = self.config.patience
        #self.empty_cache_freq = self.config.empty_cache_freq

    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, 'to'):
            data = data.to(device)
        if isinstance(data, Batch):
            data = data.to(device)

        return data
    

    def _is_main_proc(self):
        return self.local_rank == 0 or self.local_rank == -1

    def _get_version(self):
        if self.tag == 'version':
            version, pattern = -1, r'version_(\d+)'
            if os.path.exists(self.config.save_dir):
                for fname in os.listdir(self.config.save_dir):
                    ver = re.findall(pattern, fname)
                    if len(ver):
                        version = max(int(ver[0]), version)
            return version + 1
        else:
            fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
            return fn


    def _before_train_epoch_start(self):
        raise NotImplementedError()

    def _train_epoch(self, device):
        
        t_iter = tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        iteration = 0
        total_loss = 0
        total_acc = 0
        for batch in t_iter:
            #self.model.train()
            batch = self.to_device(batch, device)
            loss, acc= self.train_step(batch)
            #if torch.isnan(loss):
            #    print_log('nan detected')
            #if loss > 10.0:
            #    print_log("big loss, continue")
            #    continue #COMPRIMIZATION, need to be fixed 
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            #self.model_ema.update(self.model)
            self.global_step += 1
            #loss_list.append(loss.item() * len(batch['y'])) # for other methods 
            total_loss += loss * len(batch)
            total_acc += acc
            #self._val_epoch(device)
            del batch, loss
            iteration += 1
            #if iteration % self.empty_cache_freq == 0:
            #    torch.cuda.empty_cache()
            #    gc.collect()
            
        print_log(f'Train Loss: {total_loss.item() / len(self.train_loader.dataset):.4f}')
        print_log(f'Train Acc: {total_acc.item() / len(self.train_loader.dataset):.4f}')

    @torch.no_grad()
    def _val_epoch(self, device):

        total_loss = 0
        total_acc = 0
        self.model.eval()

        t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
        for batch in t_iter:
            batch = self.to_device(batch, device)
            loss, acc = self.val_step(batch)
            #loss_list.append(loss.cpu().item()  * len(batch[0]))
            #loss_list.append(loss.cpu().item()  * len(batch['y']))
            total_loss += loss * len(batch)
            total_acc += acc
            self.valid_global_step += 1
        # judge
        valid_metric = total_acc.item() / len(self.valid_loader.dataset)
        valid_loss = total_loss.item() / len(self.valid_loader.dataset)

        if self._is_main_proc():
            print_log(f'Val Loss: {valid_loss:.4f}')
            print_log(f'Val Acc: {valid_metric:.4f}')

        # write valid_metric
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            self.log(name, value, self.epoch)
        self.writer_buffer = {}

    @torch.no_grad()
    def _test_epoch(self, device):
        total_loss = 0
        total_acc = 0
        self.model.eval()

        t_iter = tqdm(self.test_loader) if self._is_main_proc() else self.test_loader
        for batch in t_iter:
            batch = self.to_device(batch, device)
            loss, acc = self.test_step(batch)
            #loss_list.append(loss.cpu().item()  * len(batch[0]))
            #loss_list.append(loss.cpu().item()  * len(batch['y']))
            total_loss += loss * len(batch)
            total_acc += acc
            self.valid_global_step += 1
        # judge
        if self.sched_freq == 'epoch':
            self.scheduler.step(total_loss.item() / len(self.train_loader.dataset))
            print(f"Current Learning Rate = {self.scheduler.get_last_lr()}")

        valid_metric = total_acc.item() / len(self.test_loader.dataset)
        valid_loss = total_loss.item() / len(self.test_loader.dataset)

        if self._is_main_proc():
            save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
            module_to_save = self.model.module if self.local_rank == 0 else self.model
            torch.save(module_to_save, save_path)
            
            self._maintain_topk_checkpoint(valid_metric, save_path)

            print_log(f'Test Loss: {valid_loss:.4f}')
            print_log(f'Test Acc: {valid_metric:.4f}')

        #if self._metric_better(valid_metric):
        #    self.patience = self.config.patience
        #else:
        #    self.patience -= 1
        ## write valid_metric
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            self.log(name, value, self.epoch)
        self.writer_buffer = {}
    
    def _metric_better(self, new):
        old = self.last_valid_metric
        if old is None:
            return True
        if self.config.metric_min_better:
            return new < old
        else:
            return old < new

    def _maintain_topk_checkpoint(self, valid_metric, ckpt_path):
        topk = self.config.save_topk
        if self.config.metric_min_better:
            better = lambda a, b: a < b
        else:
            better = lambda a, b: a > b
        insert_pos = len(self.topk_ckpt_map)
        for i, (metric, _) in enumerate(self.topk_ckpt_map):
            if better(valid_metric, metric):
                insert_pos = i
                break
        self.topk_ckpt_map.insert(insert_pos, (valid_metric, ckpt_path))

        # maintain topk
        if topk > 0:
            while len(self.topk_ckpt_map) > topk:
                last_ckpt_path = self.topk_ckpt_map[-1][1]
                os.remove(last_ckpt_path)
                self.topk_ckpt_map.pop()

        # save map
        self.topk_map_path = os.path.join(self.model_dir, 'topk_map.txt')
        with open(self.topk_map_path, 'w') as fout:
            for metric, path in self.topk_ckpt_map:
                fout.write(f'{metric}: {path}\n')

    def train(self, device_ids, local_rank):
        # set local rank
        self.local_rank = local_rank
        # init writer
        
        # main device
        main_device_id = local_rank if local_rank != -1 else device_ids[0]
        device = torch.device('cpu' if main_device_id == -1 else f'cuda:{main_device_id}')
        
        self.model = self.model.to(device)
        #self.model_ema.module.to(device) #not implemented
        if local_rank != -1:
            print_log(f'Using data parallel, local rank {local_rank}, all {device_ids}')
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank
            )
        else:
            print_log(f'training on {device_ids}')
        for _ in range(self.config.max_epoch):
            print_log(f'epoch{self.epoch}/{self.config.max_epoch} starts') if self._is_main_proc() else 1
            self._train_epoch(device)
            print_log(f'validating ...') if self._is_main_proc() else 1
            self._val_epoch(device)
            print_log(f'testing ...') if self._is_main_proc() else 1
            self._test_epoch(device)
            self.epoch += 1
            #if self.patience <= 0:
            #    break

    def log(self, name, value, val=False):
        if self._is_main_proc():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            if val:
                if name not in self.writer_buffer:
                    self.writer_buffer[name] = []
                self.writer_buffer[name].append(value)
            else:
                self.writer.add_scalar(name, value)

    ########## Overload these functions below ##########
    # define optimizer

    def get_optimizer(self):
        raise NotImplementedError

    def get_scheduler(self, optimizer):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError()

    def val_step(self, batch):
        raise NotImplementedError()
    
    def test_step(self, batch):
        raise NotImplementedError()


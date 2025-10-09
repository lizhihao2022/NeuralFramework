import os
import torch
import wandb
import logging

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from utils import LossRecord
from functools import partial
from models import _model_dict
from datasets import _dataset_dict


class BaseTrainer:
    def __init__(self, args):
        self.args = args
        self.model_args = args['model']
        self.data_args = args['data']
        self.optim_args = args['optimizer']
        self.scheduler_args = args['scheduler']
        self.train_args = args['train']
        self.log_args = args['log']
        
        self.logger = logging.info if self.log_args.get('log', True) else print
        self.wandb = self.log_args.get('wandb', False)
        if self.wandb:
            wandb.init(
                project=self.log_args.get('wandb_project', 'default'), 
                name=self.train_args.get('saving_name', 'experiment'),
                tags=[self.model_args.get('name', 'model'), self.data_args.get('name', 'dataset')],
                config=args)
        
        self.model_name = self.model_args['name']
        self.device = self.train_args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.logger("Building {} model".format(self.model_name))
        self.model = self.build_model(self.model_args)
        self.load_ckpt = self.train_args.get('load_ckpt', False)
        if self.load_ckpt:
            self.model.load_state_dict(torch.load(self.train_args['ckpt_path']))
            self.logger("Load model from {}".format(self.train_args['ckpt_path']))
        self.device = self.data_args['device']
        self.model = self.model.to(self.device)
        
        self.distribute = self.train_args.get('distribute', False)
        if self.distribute:
            self.distribute_mode = self.train_args.get('distribute_mode', 'DP')
            if self.distribute_mode == 'DP':
                self.device_ids = self.train_args.get('device_ids', range(torch.cuda.device_count()))
                self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
                self.logger("Using DataParallel with GPU: {}".format(self.device_ids))
            elif self.distribute_mode == 'DDP':
                dist.init_process_group(backend='nccl')
                self.local_rank = self.train_args.get('local_rank', 0)
                torch.cuda.set_device(self.local_rank)
                self.model = self.model.to(self.local_rank)
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        
        self.optimizer = self.build_optimizer(self.optim_args)
        self.scheduler = self.build_scheduler(self.scheduler_args)
        
        self.logger("Model: {}".format(self.model))
        self.logger("Model parameters: {:.2f}M".format(sum(p.numel() for p in self.model.parameters()) / 1e6))
        self.logger("Optimizer: {}".format(self.optimizer))
        self.logger("Scheduler: {}".format(self.scheduler))
        
        self.data = self.data_args['dataset']
        self.logger("Loading {} dataset".format(self.data))
        self.train_loader, self.valid_loader, self.test_loader = self.build_data(self.data_args)
        self.logger("Train dataset size: {}".format(len(self.train_loader.dataset)))
        self.logger("Valid dataset size: {}".format(len(self.valid_loader.dataset)))
        self.logger("Test dataset size: {}".format(len(self.test_loader.dataset)))
        
        self.epochs = self.train_args['epochs']
        self.eval_freq = self.train_args['eval_freq']
        self.patience = self.train_args['patience']
        
        self.saving_best = self.train_args.get('saving_best', True)
        self.saving_ckpt = self.train_args.get('saving_checkpoint', False)
        self.ckpt_freq = self.train_args.get('checkpoint_freq', 100)
        self.ckpt_max = self.train_args.get('checkpoint_max', 5)
        self.saving_path = self.train_args.get('saving_path', None)

    def get_initializer(self, name):
        if name is None:
            return None
        
        if name == 'xavier_normal':
            init_ = partial(torch.nn.init.xavier_normal_)
        elif name == 'kaiming_uniform':
            init_ = partial(torch.nn.init.kaiming_uniform_)
        elif name == 'kaiming_normal':
            init_ = partial(torch.nn.init.kaiming_normal_)
        return init_

    def build_optimizer(self, **kwargs):
        if self.optim_args['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.optim_args['lr'],
                weight_decay=self.optim_args['weight_decay'],
            )
        elif self.optim_args['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.optim_args['lr'],
                momentum=self.optim_args['momentum'],
                weight_decay=self.optim_args['weight_decay'],
            )
        elif self.optim_args['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.optim_args['lr'],
                weight_decay=self.optim_args['weight_decay'],
            )
        else:
            raise NotImplementedError("Optimizer {} not implemented".format(self.optim_args['optimizer']))
        return optimizer
    
    def build_scheduler(self, **kwargs):
        if self.scheduler_args['scheduler'] == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_args['milestones'],
                gamma=self.scheduler_args['gamma'],
            )
        elif self.scheduler_args['scheduler'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optim_args['lr'],
                div_factor=self.scheduler_args['div_factor'],
                final_div_factor=self.scheduler_args['final_div_factor'],
                pct_start=self.scheduler_args['pct_start'],
                steps_per_epoch=self.scheduler_args['steps_per_epoch'],
                epochs=self.train_args['epochs'],
            )
        elif self.scheduler_args['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_args['step_size'],
                gamma=self.scheduler_args['gamma'],
            )
        else:
            scheduler = None
            if self.scheduler_args['scheduler'] is not None:
                raise NotImplementedError("Scheduler {} not implemented".format(self.scheduler_args['scheduler']))
            
        return scheduler
    
    def build_model(self, **kwargs):
        if self.model_name not in _model_dict:
            raise NotImplementedError("Model {} not implemented".format(self.model_name))
        model = _model_dict[self.model_name](self.model_args)
        return model
    
    def build_data(self, **kwargs):
        if self.data_args['name'].lower() not in _dataset_dict:
            raise NotImplementedError("Dataset {} not implemented".format(self.data_args['name']))
        dataset = _dataset_dict[self.data_args['name'].lower()](self.data_args)
        return dataset.train_loader, dataset.valid_loader, dataset.test_loader

    def process(self, **kwargs):
        self.logger("Start training")    

        best_epoch = 0
        best_metrics = None
        counter = 0
        with tqdm(total=self.epochs) as bar:
            for epoch in range(self.epochs):
                train_loss_record = self.train(model, self.train_loader, optimizer, criterion, scheduler)
                self.logger("Epoch {} | {} | lr: {:.4f}".format(epoch, train_loss_record, optimizer.param_groups[0]["lr"]))
                if self.wandb:
                    wandb.log(train_loss_record.to_dict())
                
                if self.saving_ckpt and (epoch + 1) % self.ckpt_freq == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.cpu().state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss_record': train_loss_record.to_dict(),
                        }, os.path.join(self.saving_path, "checkpoint_{}.pth".format(epoch)))
                    model.cuda()
                    self.logger("Epoch {} | save checkpoint in {}".format(epoch, self.saving_path))
                    
                if (epoch + 1) % self.eval_freq == 0:
                    valid_loss_record = self.evaluate(model, valid_loader, criterion, split="valid")
                    self.logger("Epoch {} | {}".format(epoch, valid_loss_record))
                    valid_metrics = valid_loss_record.to_dict()
                    
                    if self.wandb:
                        wandb.log(valid_loss_record.to_dict())
                    
                    if not best_metrics or valid_metrics['valid_loss'] < best_metrics['valid_loss']:
                        counter = 0
                        best_epoch = epoch
                        best_metrics = valid_metrics
                        torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
                        model.cuda()
                        self.logger("Epoch {} | save best models in {}".format(epoch, self.saving_path))
                    elif self.patience != -1:
                        counter += 1
                        if counter >= self.patience:
                            self.logger("Early stop at epoch {}".format(epoch))
                            break
                bar.update(1)

        self.logger("Optimization Finished!")
        
        if not best_metrics:
            torch.save(model.cpu().state_dict(), os.path.join(self.saving_path, "best_model.pth"))
        else:
            model.load_state_dict(torch.load(os.path.join(self.saving_path, "best_model.pth")))
            self.logger("Load best models at epoch {} from {}".format(best_epoch, self.saving_path))        
        model.cuda()
        
        valid_loss_record = self.evaluate(model, valid_loader, criterion, split="valid")
        self.logger("Valid metrics: {}".format(valid_loss_record))
        test_loss_record = self.evaluate(model, test_loader, criterion, split="test")
        self.logger("Test metrics: {}".format(test_loss_record))
        
        if self.wandb:
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary.update(test_loss_record.to_dict())

    def train(self, model, train_loader, optimizer, criterion, scheduler=None, **kwargs):
        loss_record = LossRecord(["train_loss"])
        model.cuda()
        model.train()
        for x, y in train_loader:
            x = x.to('cuda')
            y = y.to('cuda')
            y_pred = model(x).reshape(y.shape)
            data_loss = criterion(y_pred, y)
            loss = data_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_record.update({"train_loss": loss.sum().item()}, n=y_pred.shape[0])

        if scheduler is not None:
            scheduler.step()
        return loss_record
    
    def evaluate(self, model, eval_loader, criterion, split="valid", **kwargs):
        loss_record = LossRecord(["{}_loss".format(split)])
        model.eval()
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                y_pred = model(x).reshape(y.shape)
                data_loss = criterion(y_pred, y)
                loss = data_loss
                loss_record.update({"{}_loss".format(split): loss.sum().item()}, n=y_pred.shape[0])
        return loss_record

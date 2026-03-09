# here put the import lib
import os

import torch
from tqdm import trange
from earlystop import EarlyStoppingNew
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from transformers import get_cosine_schedule_with_warmup
class CosineAnnealingWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr=0, last_epoch=-1):
        
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        super(CosineAnnealingWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        
        step = self.last_epoch
        if step < self.warmup_steps:
            
            warmup_lr = (self.base_lrs[0] - self.min_lr) * step / self.warmup_steps + self.min_lr
            return [warmup_lr] * len(self.optimizer.param_groups)
        else:
            
            cosine_lr = self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * \
                        (1 + np.cos(np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            return [cosine_lr] * len(self.optimizer.param_groups)
    
    def step(self, epoch=None):
        
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]
        
class Trainer(object):

    def __init__(self, args, model, train_dataloader, 
valid_dataloader, logger, writer):
    
        self.logger = logger
        self.writer = writer
        self.args = args
        self.start_epoch = 0    
        self.model = model.to(self.args.gpu)
        if args.distributed:
            dist.barrier()
            print(f"Initializing DDP on GPU {args.gpu}...")
            self.model = DDP(self.model, device_ids=[args.gpu])
            print("DDP initialized successfully")
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self._set_optimizer()
        self._set_scheduler()
        self._set_stopper()

        
        if args.local_rank == 0:
            print(f'Iter {args.local_rank} is ready to train')

    
    def _create_model(self):
        '''create your model'''
        pass
        
    

    def _load_pretrained_model(self):

        self.logger.info("Loading the trained model for keep on training ... ")
        checkpoint_path = os.path.join(self.args.keepon_path, 'pytorch_model.bin')

        model_dict = self.model.state_dict()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pretrained_dict = checkpoint['state_dict']

        
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        
        self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        self.model.load_state_dict(model_dict)  
        self.optimizer.load_state_dict(checkpoint['optimizer']) 
        self.scheduler.load_state_dict(checkpoint['scheduler']) 
        self.start_epoch = checkpoint['epoch']  

    
    def _set_optimizer(self):
        
        if self.args.optim == "adamw_torch":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.optim == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                            lr=self.args.learning_rate,
                                            weight_decay=self.args.weight_decay,
                                            )

    
    def _set_scheduler(self):
        if self.args.lr_scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=self.args.lr_dc_step,
                                                             gamma=self.args.lr_dc)
        elif self.args.lr_scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                        T_max=self.args.epochs,
                                                                        eta_min=5e-4)
        elif self.args.lr_scheduler_type == 'warmup_cosine':
            total_steps = len(self.train_loader) * self.args.epochs
            warmup_steps = int(self.args.warmup_ratio * total_steps)  
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                            num_warmup_steps=warmup_steps,
                                                            num_training_steps=total_steps,
                                                            )

    def _set_stopper(self):

        self.stopper = EarlyStoppingNew(patience=self.args.patience, 
                                     verbose=False,
                                     path=self.args.output_dir,
                                     trace_func=self.logger)


    def _train_one_epoch(self, epoch):

        return NotImplementedError
    

    def _prepare_train_inputs(self, data):
        """Prepare the inputs as a dict for training"""
        assert len(self.generator.train_dataset.var_name) == len(data)
        inputs = {}
        for i, var_name in enumerate(self.generator.train_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs
    

    def _prepare_eval_inputs(self, data):
        """Prepare the inputs as a dict for evaluation"""
        inputs = {}
        assert len(self.generator.eval_dataset.var_name) == len(data)
        for i, var_name in enumerate(self.generator.eval_dataset.var_name):
            inputs[var_name] = data[i]

        return inputs


    def eval(self, epoch=0, test=False):

        return NotImplementedError


    def train(self):

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.per_device_batch_size)
        res_list = []
        train_time = []

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.epochs), desc="Epoch"):
            if self.args.distributed:
                self.train_loader.sampler.set_epoch(epoch)
                
            t = self._train_one_epoch(epoch)
            
            train_time.append(t)
            
            if (epoch % 1) == 0:
                if self.args.distributed:
                    if  self.args.local_rank == 0:
                        
                        eval_loss = self.eval(epoch=epoch)
                        res_list.append(eval_loss)
                        if self.args.distributed:
                            dist.barrier()
                        if self.args.local_rank == 0:
                            print('\nEpoch: %d, eval_loss: %.5f\n' % (epoch, eval_loss))
                        
                        torch.save({'epoch': epoch,
                                    'state_dict': model_to_save.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'scheduler': self.scheduler.state_dict()}
                                    ,os.path.join(self.args.output_dir, "pytorch_model.bin"))
                        print(os.path.join(self.args.output_dir, "pytorch_model.bin"))

                else:
                    eval_loss = self.eval(epoch=epoch)
                    torch.save({'epoch': epoch,
                                    'state_dict': model_to_save.state_dict(),
                                    'optimizer': self.optimizer.state_dict(),
                                    'scheduler': self.scheduler.state_dict()}
                                    ,os.path.join(self.args.output_dir, "pytorch_model.bin"))
                    
                    print('\nEpoch: %d, eval_loss: %.5f\n' % (epoch, eval_loss))
            if self.args.dataset == 'sports':
                end_epoch = 10
            else:
                end_epoch = 5
            
            if epoch == end_epoch:
                break        

    




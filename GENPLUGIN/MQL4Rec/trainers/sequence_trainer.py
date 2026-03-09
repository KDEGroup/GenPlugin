import os
import time
import torch
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn
import shutil
import torch.nn.functional as F

class SeqTrainer(Trainer):

    def __init__(self, args, model, train_dataloader, valid_dataloader, logger, writer):
        super().__init__(args, model, train_dataloader, valid_dataloader, logger, writer)

    
    def _train_one_epoch(self, epoch):
        tr_loss = 0
        steps = 0

        self.model.train()
        
        for batch in self.train_loader:
            if self.args.distributed:
                loss, id_mean, text_mean = self.model.module(batch, epoch)
            else:
                loss, id_mean, text_mean = self.model(batch, epoch)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            
            if self.args.distributed:
                dist.barrier()

            tr_loss += loss.item()
            steps += 1

            
                
        
        if self.args.local_rank == 0:
            avg_train_loss = tr_loss / steps
            self.writer.add_scalar('train/loss', avg_train_loss, epoch)
            return avg_train_loss

    def eval(self, epoch=0, train_mode='train', data_mode='eval', test=False):
        print('')
        self.logger.info("\n----------------------------------")
        self.logger.info(f"********** Epoch: {epoch} eval **********")
        
        
        if test:
            self.logger.info("********** Running test **********")
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        else:
            test_loader = self.valid_loader

        
        self.model.eval()
        total_loss = 0
        index = 0

        for batch in tqdm(test_loader, desc='Evaluating' if not test else 'Testing'):
            with torch.no_grad():
                if train_mode == 'train' or train_mode == 'eval':
                    if self.args.distributed:
                        loss = self.model.module.evaluate(batch, index, train_mode)
                    else:
                        loss = self.model.evaluate(batch, index, train_mode)
                elif train_mode == 'rag':
                    
                    output_dir = f"../../MQL4Rec-RAR/{self.args.dataset}/{data_mode}"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    if self.args.distributed:
                        loss = self.model.module.evaluate(batch, index, train_mode, data_mode)
                    else:
                        loss = self.model.evaluate(batch, index, train_mode, data_mode)
                total_loss += loss.item()
                index += 1

        avg_loss = total_loss / len(test_loader)

        
        return avg_loss

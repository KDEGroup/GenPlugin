
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn as nn
class SeqTrainer(Trainer):

    def __init__(self, args, model, train_dataloader,
valid_dataloader, logger, writer):
        super().__init__(args, model, train_dataloader, valid_dataloader,logger, writer)
        
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        train_time = []
        steps = 0
        self.model.train()
        

        for batch in self.train_loader:
            if self.args.distributed:
                loss = self.model.module(batch, epoch)
            else:
                loss = self.model(batch, epoch)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            if self.args.distributed:
                dist.barrier()         
            tr_loss += loss.item()
            steps += 1
        torch.cuda.empty_cache()
        if self.args.local_rank == 0:
            self.writer.add_scalar('train/loss', tr_loss / steps, epoch)




    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        if self.args.distributed:
            if self.args.local_rank == 0:
                self.model.eval()
                total_loss = 0
                
                for batch in tqdm(test_loader, desc=desc):
                    with torch.no_grad():
                        loss = self.model.module.evaluate(batch, index)
                        total_loss += loss.item()
                        index += 1
                        torch.cuda.empty_cache()
                avg_loss = total_loss / len(test_loader)
                
                return avg_loss
        else:
            self.model.eval()
            total_loss = 0
            index = 0
            for batch in tqdm(test_loader, desc=desc):
                with torch.no_grad():
                    loss = self.model.evaluate(batch, index)
                    total_loss += loss.item()
                    index += 1
                    torch.cuda.empty_cache()
            avg_loss = total_loss / len(test_loader)
                    
            return avg_loss


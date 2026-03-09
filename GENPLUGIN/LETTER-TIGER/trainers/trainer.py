import os
import torch
from tqdm import trange
from earlystop import EarlyStoppingNew
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from transformers import get_cosine_schedule_with_warmup
from typing import Dict, List, Optional, Tuple
import time
from contextlib import contextmanager
class Trainer:
    def __init__(self, args, model, train_dataloader, valid_dataloader, logger, writer):
        self.args = args
        self.logger = logger
        self.writer = writer
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.start_epoch = 0

        
        if args.distributed:
            self._setup_distributed()
        
        
        self._set_optimizer()
        self._set_scheduler()
        self._set_stopper()
        
        
        
        if args.local_rank == 0:
            self.logger.info(f"Trainer initialized on rank {args.local_rank}")

    def _setup_distributed(self):
        dist.barrier()
        self.logger.info(f"Initializing DDP on GPU {self.args.gpu}")
        self.model = DDP(self.model, device_ids=[self.args.gpu], 
                        find_unused_parameters=True)
        self.logger.info("DDP initialized successfully")

    def _set_optimizer(self):
        optimizers = {
            "adamw_torch": lambda: torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            ),
            "adam": lambda: torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        }
        self.optimizer = optimizers[self.args.optim]()
        with torch.cuda.device(self.args.gpu):
            torch.cuda.empty_cache()

    def _set_scheduler(self):
        total_steps = len(self.train_loader) * self.args.epochs
        warmup_steps = int(self.args.warmup_ratio * total_steps)
        
        schedulers = {
            'step': lambda: torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.args.lr_dc_step,
                gamma=self.args.lr_dc
            ),
            'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs,
                eta_min=5e-4
            ),
            'warmup_cosine': lambda: get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            ),
        }
        self.scheduler = schedulers[self.args.lr_scheduler_type]()

    def _set_stopper(self):
        self.stopper = EarlyStoppingNew(
            patience=self.args.patience,
            verbose=False,
            path=self.args.output_dir,
            trace_func=self.logger
        )

    def _load_pretrained_model(self):
        self.logger.info("Loading pretrained model for continued training...")
        checkpoint_path = os.path.join(self.args.keepon_path, 'pytorch_model.bin')
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_dict = self.model.state_dict()
        
        
        pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.start_epoch = checkpoint['epoch']
        
        self.logger.info(f"Loaded {len(pretrained_dict)} parameters from checkpoint")

    @contextmanager
    def _training_context(self):
        """Context manager for training setup and cleanup"""
        self.model.train()
        with torch.cuda.device(self.args.gpu):
            torch.cuda.empty_cache()
        try:
            yield
        finally:
            with torch.cuda.device(self.args.gpu):
                torch.cuda.empty_cache()

    def train(self):
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        self.logger.info("\n********** Starting Training **********")
        self.logger.info(f"Batch size: {self.args.per_device_batch_size}")
        
        results = []
        best_loss = float('inf')
        
        with trange(self.start_epoch, self.start_epoch + self.args.epochs, 
                   desc="Epoch") as pbar:
            for epoch in pbar:
                if self.args.distributed:
                    self.train_loader.sampler.set_epoch(epoch)
                if self.args.train_mode == 'train':
                    with self._training_context():
                        train_time = self._train_one_epoch(epoch)
                    if epoch % 1 == 0 and self.args.local_rank == 0:
                        eval_loss = self.eval(epoch=epoch, 
                                           train_mode=self.args.train_mode,
                                           data_mode=self.args.data_mode)
                        results.append(eval_loss)
                        
                        if self.args.distributed:
                            dist.barrier()
                        
                        pbar.set_description(f"Epoch {epoch} - Eval Loss: {eval_loss:.5f}")
                        self.stopper(-eval_loss, epoch, model_to_save, 
                                  self.optimizer, self.scheduler)
                        
                        if eval_loss < best_loss:
                            best_loss = eval_loss
                            best_epoch = epoch
                        
                        if self.stopper.early_stop:
                            break
                
                elif self.args.train_mode == 'rag':
                    eval_loss = self.eval(epoch=epoch,
                                        train_mode=self.args.train_mode,
                                        data_mode=self.args.data_mode)
                    self.logger.info(f"Eval Loss: {eval_loss:.5f}")
                    return
                
                self.scheduler.step()

        if self.args.local_rank == 0:
            self.logger.info(f"Best epoch: {best_epoch}")
            self.logger.info(f"Best eval loss: {best_loss:.5f}")
        
        if self.args.distributed:
            dist.barrier()

    def _train_one_epoch(self, epoch: int) -> float:
        raise NotImplementedError

    def eval(self, epoch: int = 0, test: bool = False, 
            train_mode: str = 'train', data_mode: str = 'eval') -> float:
        raise NotImplementedError

    def _prepare_inputs(self, data: List, dataset_type: str) -> Dict:
        """Generic input preparation method"""
        dataset = getattr(self.generator, f"{dataset_type}_dataset")
        assert len(dataset.var_name) == len(data)
        return dict(zip(dataset.var_name, data))

    def get_model_param_num(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total - trainable, trainable

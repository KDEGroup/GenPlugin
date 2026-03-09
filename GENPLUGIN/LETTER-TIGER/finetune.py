import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
import sys
from typing import List
from transformers import EarlyStoppingCallback
from torch.nn.parallel import DistributedDataParallel as DDP                                                                        
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from modeling import LETTER, final_model
# import wandb
from utils import *
from collator import Collator
from logger import Logger
from trainers.sequence_trainer import SeqTrainer
def train(gpu, args):
    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    parser.now_str = log_manager.get_now_str()
    
    args.gpu = gpu
    args.rank = gpu
    print(f'gpu: {gpu}')
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')
        
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    


    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model,
        model_max_length=512,
    )
    args.deepspeed = None
    gradient_checkpointing= False


    train_data, valid_data = load_datasets(args)
    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    if args.local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)


    collator = Collator(args, tokenizer)
    
    if args.distributed:
        sampler_train = DistributedSampler(train_data)
    else:
        sampler_train = None

    train_loader = DataLoader(
        train_data,
        batch_size=args.per_device_batch_size,
        shuffle=(sampler_train is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler_train,
        collate_fn=collator,
        drop_last=False)
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False)
    
    if args.distributed:
        dist.barrier()
        
    model = final_model(config, args).to(args.gpu)
    model.id_model.resize_token_embeddings(len(tokenizer))
    model.text_model.resize_token_embeddings(len(tokenizer))
    
    for name, param in model.named_parameters():
        if 'text_emb' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    if args.train_mode == 'rag':
        model_state_dict = torch.load(f'./ckpt/{args.dataset}-rag/pytorch_model.bin')
        model.load_state_dict(model_state_dict['state_dict'], strict=False)
    trainer = SeqTrainer(args, model, train_loader, valid_loader, logger, writer)
    
    return trainer


    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_logger_args(parser)
    args = parser.parse_args()
    args.distributed = False
    if args.local_rank == 0:
        print(args, '\n')
        
    if not args.distributed:
        seq_train = train(args.gpu, args)
    else:
       seq_train = train(args.local_rank, args)  

    seq_train.train()
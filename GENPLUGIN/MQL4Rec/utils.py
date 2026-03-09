import json
import logging
import os
import random
import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data import SeqRecDataset, ItemImageDataset, FusionSeqRecDataset

def parse_global_args(parser):


    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--base_model", type=str, default="../LETTER-TIGER/ckpt/TIGER",help="basic model path")

    parser.add_argument("--output_dir", type=str, default="./ckpt",
                        help="The output directory")
    parser.add_argument("--model_name", type=str, default="LETTER-TIGER")
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="../data",
                        help="data directory")
    parser.add_argument("--tasks", type=str, default="seqrec",
                        help="Downstream tasks, separate by comma")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".llamaindex-sk4-sk.json", help="the item indices file")
    parser.add_argument("--image_index_file", type=str, default=".index.json", help="the item indices file")
    parser.add_argument("--fg_image_index_file", type=str, default=".index.json", help="the item indices file")
    
    parser.add_argument("--prompt_num", type=int, default=4,
                            help="soft prompt num")
    
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of items in history sequence, -1 means no limit")
    parser.add_argument("--add_prefix", action="store_true", default=False,
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=", ", help="The separator used for history")
    parser.add_argument("--only_train_response", action="store_true", default=False,
                        help="whether only train on responses")

    parser.add_argument("--train_prompt_sample_num", type=str, default="1",
                        help="the number of sampling prompts for each task")
    parser.add_argument("--train_data_sample_num", type=str, default="-1",
                        help="the number of sampling prompts for each task")

    
    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=True,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=2,
                        help="the number of sampling validation sequential recommendation prompts")
    
    parser.add_argument("--type", type=str, default="all")
    
    return parser

def parse_train_args(parser):
    parser.add_argument("--distributed", default=True, help="whether use distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.005)
    parser.add_argument("--lr_scheduler_type", type=str, default="warmup_cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")
    parser.add_argument("--wandb_run_name", type=str, default="default")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=20, help="The patience for early stopping")
    parser.add_argument("--lr_dc_step",default=1000,type=int,help='every n step, decrease the lr')
    parser.add_argument("--lr_dc",default=0,type=float,
                        help='how many learning rate to decrease')
    
    parser.add_argument("--keepon", type=bool, default="False",help="keep on training")
    parser.add_argument("--keepon_path", type=str, default="pretrain",help="keep on training")
    parser.add_argument("--valid_task", type=str, default="SeqRec")
    parser.add_argument("--train_mode", type=str, default="train", help="train or rag")
    parser.add_argument("--data_mode", type=str, default="val", help="eval or test or train")
    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str,
                        default="./ckpt",
                        help="The checkpoint path")
    parser.add_argument("--filter_items", action="store_true", default=True,
                        help="whether filter illegal items")

    parser.add_argument("--results_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")
    parser.add_argument("--save_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")
    
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--test_task", type=str, default="SeqRec")
    parser.add_argument("--test_mode", type=str, default="test", help="test or rag")
    parser.add_argument("--data_mode", type=str, default="val", help="eval or test or train")
    return parser

def parse_logger_args(parser):
    parser.add_argument("--log_path", type=str, default="./logs",
                        help="log path")
    parser.add_argument("--log_name", type=str, default="test.log",
                        help="log name")
    parser.add_argument("--log", type=str, default="False")
    parser.add_argument("--demo", type=str, default="True")
    return parser

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)


def load_datasets(args):

    tasks = args.tasks.split(",")

    
    
    train_prompt_sample_num = [1] * len(tasks)
    train_data_sample_num = [-1] * len(tasks)

    train_datasets = []

    print(tasks)
    train_datasets = []
    for task, prompt_sample_num,data_sample_num in zip(tasks,train_prompt_sample_num,train_data_sample_num):
        if task.lower() == "seqrec":
            dataset = SeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
        elif task.lower() == "seqimage":
            dataset = SeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
            
        elif task.lower() == "item2image" or task.lower() == "image2item":
            dataset = ItemImageDataset(args, task=task.lower(), prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)

        elif task.lower() == "seqitem2image" or task.lower() == "seqimage2item" or task.lower() == "fusionseqrec":
            dataset = FusionSeqRecDataset(args, task=task.lower(), mode="train", prompt_sample_num=prompt_sample_num, sample_num=data_sample_num)
            
        else:
            raise NotImplementedError
        print("task:", task)
        print("train dataset size:", len(dataset))
        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_datasets = []
    valid_tasks = args.valid_task.lower().split(',')
    for valid_task in valid_tasks:
        dataset = SeqRecDataset(args, task=valid_task, mode="valid", data_mode=args.data_mode, prompt_sample_num=args.valid_prompt_sample_num)
        valid_datasets.append(dataset)
    valid_data = ConcatDataset(valid_datasets)
    return train_data, valid_data

def load_test_dataset(args):

    if args.test_task.lower() == "seqrec" or args.test_task.lower() == "seqimage":
        
        test_data = SeqRecDataset(args,task=args.test_task.lower(), mode="test", data_mode=args.data_mode, sample_num=args.sample_num)
    else:
        raise NotImplementedError

    return test_data

def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out

    return prefix_allowed_tokens

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data
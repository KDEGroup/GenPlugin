import json
import logging
import os
import random
import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data import SeqRecDataset

def parse_global_args(parser):


    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--base_model", type=str, default="../LETTER-TIGER/ckpt/TIGER",help="basic model path")

    parser.add_argument("--output_dir", type=str, default="./ckpt",
                        help="The output directory")
    parser.add_argument("--model_name", type=str, default="LETTER-TIGER")
    parser.add_argument("--model_type", type=str, default="tiger")
    parser.add_argument("--task", type=str, default="seqrec",
                        help="Downstream tasks, separate by comma")
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="../data",
                        help="data directory")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Dataset name")
    parser.add_argument("--index_file", type=str, default=".llamaindex-sk4-sk.json", help="the item indices file")
    parser.add_argument("--image_index_file", type=str, default=".index.json", help="the item indices file")
    parser.add_argument("--prompt_num", type=int, default=4,
                            help="soft prompt num")
    # arguments related to sequential task
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

    # arguments related for evaluation
    parser.add_argument("--valid_prompt_id", type=int, default=0,
                        help="The prompt used for validation")
    parser.add_argument("--sample_valid", action="store_true", default=True,
                        help="use sampled prompt for validation")
    parser.add_argument("--valid_prompt_sample_num", type=int, default=2,
                        help="the number of sampling validation sequential recommendation prompts")
    
    parser.add_argument("--type", type=str, default="all")
    parser.add_argument("--train_mode", type=str, default="rag")
    return parser

def parse_train_args(parser):
    parser.add_argument("--distributed",type=bool, default=False, help="whether use distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--optim", type=str, default="adamw_torch", help='The name of the optimizer')
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.02)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="warmup_cosine")
    parser.add_argument("--save_and_eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--fp16",  action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")
    parser.add_argument("--wandb_run_name", type=str, default="default")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5, help="The patience for early stopping")
    parser.add_argument("--lr_dc_step",default=1000,type=int,help='every n step, decrease the lr')
    parser.add_argument("--lr_dc",default=0,type=float,
                        help='how many learning rate to decrease')
    
    parser.add_argument("--keepon", type=bool, default="False",help="keep on training")
    parser.add_argument("--keepon_path", type=str, default="pretrain",help="keep on training")
    
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

    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID when testing with single GPU")
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--save_file", type=str,
                    default="./results/test-ddp.json",
                    help="result output path")
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

    task = args.task
    train_datasets = []
    if task.lower() == "seqrec":
        dataset = SeqRecDataset(args,task=task.lower(), mode="train")
    elif task.lower() == "seqimage":
        dataset = SeqRecDataset(args,task=task.lower(), mode="train")
    else:
        raise NotImplementedError
    train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)

    valid_data = SeqRecDataset(args,task.lower(),"valid")

    return train_data, valid_data

def load_test_dataset(args):

    if args.task.lower() == "seqrec":
        test_data = SeqRecDataset(args,task=args.task.lower(), mode="test", sample_num=args.sample_num)
    elif args.task.lower() == "seqimage":
        test_data = SeqRecDataset(args,task=args.task.lower(), mode="test", sample_num=args.sample_num)
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

from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
@dataclass
class LETTER_Seq2SeqLMOutput(ModelOutput):


    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_mask: Optional[torch.Tensor] = None
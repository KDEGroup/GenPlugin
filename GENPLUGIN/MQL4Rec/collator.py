import torch
import copy
import argparse
from dataclasses import dataclass
import numpy as np
import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration



class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        end_dict = {'Beauty': 12101, 'Instruments': 6250, 'Arts': 9416, 'toys':11924, 'beauty': 12101, 'sports': 18357, 'yelp': 20033, 'toys':11924}
        self.end_token_id = end_dict[args.dataset]
        

    def __call__(self, batch):
        new_batch = dict()
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]
        index = [d['index'] for d in batch]
        id_type = [d['id_type'] for d in batch]
        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100
        new_batch['lm_inputs'] = inputs
        
        max_len = max([len(d["item_idx"]) for d in batch])
        text_input_ids = []
        text_attention_mask = []
        end_token_id = self.end_token_id
        
        for d in batch:
            item_idx = d["item_idx"]
            
            
            padded_item_idx = item_idx+  [end_token_id] +  [end_token_id] * (max_len - len(item_idx))
            
            
            mask = [1] * (len(item_idx) + 1) + [0] * (max_len - len(item_idx))
            
            text_input_ids.append(padded_item_idx)
            text_attention_mask.append(mask)

        new_batch['text_input_ids'] = torch.tensor(text_input_ids)
        new_batch['text_attention_mask'] = torch.tensor(text_attention_mask)
        
        new_batch['item_id'] = torch.tensor([d['item_id'] for d in batch])
        new_batch['index'] = index
        new_batch['id_type'] = id_type
        return new_batch



class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        end_dict = {'Beauty': 12101, 'Instruments': 6250, 'Arts': 9416, 'toys':11924, 'beauty': 12101, 'sports': 18357, 'yelp': 20033, 'toys':11924}
        self.end_token_id = end_dict[args.dataset]
    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        max_len = max([len(d["item_idx"]) for d in batch])
        text_input_ids = []
        text_attention_mask = []
        end_token_id = self.end_token_id
        uid = [d['index'] for d in batch]
        for d in batch:
            item_idx = d["item_idx"]
            
            
            padded_item_idx = item_idx +  [end_token_id] +  [end_token_id] * (max_len - len(item_idx))
            
            
            mask = [1] * (len(item_idx) + 1) + [0] * (max_len - len(item_idx))
            
            text_input_ids.append(padded_item_idx)
            text_attention_mask.append(mask)

        text_input_ids = torch.tensor(text_input_ids, dtype=torch.long)
        text_attention_mask = torch.tensor(text_attention_mask, dtype=torch.long)
        
        
        return (inputs, targets, text_input_ids, text_attention_mask, uid)


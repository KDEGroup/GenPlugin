import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import torch.distributed as dist
import logging
import re
import pdb
import json
import numpy as np
from transformers import T5Tokenizer


class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.image_index_file = args.image_index_file
        self.add_prefix = args.add_prefix

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None
        self.image_indices = None
        self.user_type = args.type

    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.image_index_file), 'r') as f:
            image_indices = json.load(f)
            
        self.image_indices = {}
        for k, v in image_indices.items():
            if random.random() > 0.5:
                self.image_indices[k] = v
    def get_new_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
                
        if self.image_indices is not None:
            for index in self.image_indices.values():
                for token in index:
                    self.new_tokens.add(token)
                    
                    
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens
    def get_all_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens
        
        if self.args.tasks == 'seqrec':
            prefix_list = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>"]
            
        elif self.args.tasks == 'seqimage':
            prefix_list = ["<A_{}>","<B_{}>","<C_{}>","<D_{}>"]
            
        else:
            prefix_list = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>", "<A_{}>","<B_{}>","<C_{}>","<D_{}>"]
            
        new_tokens = set()
        
        for prefix in prefix_list:
            for i in range(self.args.code_num):
                token = prefix.format(i)
                new_tokens.add(token)
                
        self.new_tokens = sorted(list(new_tokens))
        
        

        return self.new_tokens
    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_all_items_v2(self):
        if self.all_items is not None:
            return self.all_items

        self.all_items = []
        for index in self.indices.values():
            self.all_items.append("".join(index))

        return self.all_items       
    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][0]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, task='seqrec', mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.task = task
        self.soft_prompt = args.soft_prompts[self.task]

        
        self._load_data()
        self._remap_items()
        
        
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        elif self.mode == 'test_ranking':
            self.inter_data = self._process_test_data_ids()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        if self.task == 'seqrec':
            with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
                self.indices = json.load(f)
        elif self.task == 'seqimage':
            with open(os.path.join(self.data_path, self.dataset + self.image_index_file), 'r') as f:
                self.indices = json.load(f)
        
    def _remap_items(self):

        if self.mode != "test":
            self.remapped_inters = dict()
            for uid, items in self.inters.items():
                new_items = ["".join(self.indices[str(i)]) for i in items]
                self.remapped_inters[uid] = new_items
        else:
            
            if self.user_type == "all":
                self.remapped_inters = dict()
                for uid, items in self.inters.items():
                    new_items = ["".join(self.indices[str(i)]) for i in items]
                    self.remapped_inters[uid] = new_items
            else:
                raise NotImplementedError


    def _process_train_data(self):

        inter_data = []
        for uid  in self.remapped_inters:
            items = self.remapped_inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                
                one_data["item"] = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                if self.add_prefix:
                    history = [str(k+1) + ". " + item_idx for k, item_idx in enumerate(history)]
                
                one_data["inters"] = "".join(history)
                
                one_data['item_idx'] = self.inters[uid][:i]
                if self.max_his_len > 0:
                    one_data['item_idx'] = one_data['item_idx'][-self.max_his_len:]
                one_data['item_id'] = self.inters[uid][i]
                inter_data.append(one_data)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data['item_idx'] = self.inters[uid][:-2]
            if self.max_his_len > 0:
                one_data['item_idx'] = one_data['item_idx'][-self.max_his_len:]
            
            
            one_data["inters"] = "".join(history)
            
            
            one_data['item_id'] = self.inters[uid][-2]
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            
            items = self.remapped_inters[uid]
            one_data = dict()
            
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            if self.add_prefix:
                history = [str(k + 1) + ". " + item_idx for k, item_idx in enumerate(history)]
            one_data['item_idx'] = self.inters[uid][:-2]
            if self.max_his_len > 0:
                one_data['item_idx'] = one_data['item_idx'][-self.max_his_len:]
            
            
            one_data["inters"] = "".join(history)
            
            user_id = uid
            one_data['user_id'] =int(user_id)
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data     
    

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]
        if self.mode != 'test':
            uid = index
        else:
            uid = d['user_id']
        return dict(input_ids=self.soft_prompt + d["inters"], labels=d["item"], \
            item_nums=d['item_nums'] if 'item_nums' in d else None, \
            item_idx=d['item_idx'] if 'item_idx' in d else None, \
            item_id=d['item_id'] if 'item_id' in d else None, \
            index=uid)
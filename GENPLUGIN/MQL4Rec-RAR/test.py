import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import random
import sys
from typing import List

import torch
import transformers

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration
from modeling import LETTER, final_model
from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie


def test(args):

    set_seed(args.seed)
    print(vars(args))

    device_map = {"": args.gpu_id}
    device = torch.device("cuda",args.gpu_id)

    
    config = T5Config.from_pretrained(args.ckpt_path)
    tokenizer = T5Tokenizer.from_pretrained(
        args.ckpt_path,
        model_max_length=512,
    )
    
    tasks = ['seqrec','seqimage']
    soft_prompts = {}
    for i, task in enumerate(tasks):
        if task == 'fgfusionseqrec':
            token_ids = list(range(100 * (i + 1), 100 * (i + 1) + args.prompt_num * 4))
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            prompt = []
            for k in range(4):
                p = ''.join(tokens[args.prompt_num * k : args.prompt_num * (k+1)])
                prompt.append(p)
            
        else:
            token_ids = list(range(100 * (i + 1), 100 * (i + 1) + args.prompt_num))
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            prompt = ''.join(tokens)
            

            
        soft_prompts[task] = prompt
        
    args.soft_prompts = soft_prompts

    
    model = final_model(config,args)
    print(args.ckpt_path)
    
    model_state_dict = torch.load(args.ckpt_path+'/pytorch_model.bin')
    print(model_state_dict.keys())
    model.load_state_dict(model_state_dict['state_dict'], strict=False)
    model.eval()
    model.to(device)
    prompt_ids = [0]

    test_data = load_test_dataset(args)
    print(test_data[0])
    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    candidate_trie = Trie(
        [
            [0] + tokenizer.encode(candidate)
            for candidate in all_items
        ]
    )
    prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)


    print("data num:", len(test_data))

    model.eval()
    all_outputs = []
    all_scores = []
    all_targets = []
    all_users = []
    
    save_dict = {}
    metrics = args.metrics.split(",")
    all_prompt_results = []
    with torch.no_grad():
        for prompt_id in prompt_ids:

            
            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for step, batch in enumerate(tqdm(test_loader)):
                
                targets = batch[1]
                users = batch[4]
                total += len(targets)
                if step == 0:
                    
                    print(targets)

                output = model.predict(batch, prefix_allowed_tokens=prefix_allowed_tokens)
                output_ids = output["sequences"]
                scores = output["sequences_scores"]

                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                all_outputs.extend(output)
                all_scores.extend(scores.cpu().tolist())
                all_targets.extend(targets)
                all_users.extend(users)
                
                save_dict['all_outputs'] = all_outputs
                save_dict['all_scores'] = all_scores
                save_dict['all_targets'] = all_targets
                save_dict['all_users'] = all_users
                
                topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                            all_items=all_items if args.filter_items else None)

                batch_metrics_res = get_metrics_results(topk_res, metrics)
                

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                
                temp={}
                for m in metrics_results:
                    temp[m] = metrics_results[m] / total
                print(temp)
            
            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total
            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")
    with open(args.save_file, "w") as f:
        json.dump(save_dict, f, indent=4)
    mean_results = {}
    min_results = {}
    max_results = {}

    for m in metrics:
        all_res = [_[m] for _ in all_prompt_results]
        mean_results[m] = sum(all_res)/len(all_res)
        min_results[m] = min(all_res)
        max_results[m] = max(all_res)

    print("======================================================")
    print("Mean results: ", mean_results)
    print("Min results: ", min_results)
    print("Max results: ", max_results)
    print("======================================================")


    save_data={}
    save_data["test_prompt_ids"] = args.test_prompt_ids
    save_data["mean_results"] = mean_results
    save_data["min_results"] = min_results
    save_data["max_results"] = max_results
    save_data["all_prompt_results"] = all_prompt_results

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()

    test(args)

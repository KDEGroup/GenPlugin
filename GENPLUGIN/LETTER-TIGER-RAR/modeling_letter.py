from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration
)
import torch
from torch import nn
import copy
import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer
from sklearn.metrics.pairwise import cosine_similarity

import math
from torch.nn.init import xavier_normal_
import os
from math import sqrt
import faiss
import time
from utils import LETTER_Seq2SeqLMOutput
import random

      
class LETTER(T5ForConditionalGeneration):

    def __init__(self, config: T5Config, dataset, model_type):

        super().__init__(config)

        self.temperature = 1.0
        self.test_sparse_index = json.load(open(f"./rag_need/{model_type}/{dataset}/test/rag_user_index.json"))
        self.test_sas_index = json.load(open(f"./rag_need/{model_type}/{dataset}/test/item_retrival.json"))
        self.test_sparse_index = np.array(self.test_sparse_index)
        self.test_sas_index = np.array(self.test_sas_index)
        self.user_mean_emb = np.load(f"./rag_need/{model_type}/{dataset}/test/user_emb_mean.npy")
        self.text_user_mean_emb = np.load(f"./rag_need/{model_type}/{dataset}/test/text_user_emb_mean.npy")
        self.test_rerank = json.load(open(f"./rag_need/{model_type}/{dataset}/test/reranked_user_index.json"))
        self.test_rerank = np.array(self.test_rerank)
        
    def set_hyper(self,temperature):
        self.temperature = temperature


    def ranking_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits/self.temperature
            loss_fct = CrossEntropyLoss(reduction='none',ignore_index=-100)
            
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        else:
            loss = 0
        return loss.mean()

    
    def total_loss(self, lm_logits, labels, decoder_input_ids):
        loss = self.ranking_loss(lm_logits, labels)             
        return loss
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, **kwargs
        )

        
        if "text_encoder_outputs" in kwargs:
            model_inputs["text_encoder_outputs"] = kwargs["text_encoder_outputs"]
        if "text_attention_mask" in kwargs:
            model_inputs["text_attention_mask"] = kwargs["text_attention_mask"]
        if "user_rag_emb" in kwargs:
            model_inputs["user_rag_emb"] = kwargs["user_rag_emb"]
        return model_inputs
    def forward(
        self,
        input_ids=None,
        whole_word_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cross_attn_head_mask = None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=False,

        return_hidden_state=False,
        text_encoder_outputs=None, 
        text_attention_mask=None,
        user_rag_emb = None,
        **kwargs,
    ):
        r"""

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        
        if encoder_outputs is None:
            
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        user_emb = hidden_states.mean(dim=1)
        if user_rag_emb is not None and text_encoder_outputs is None:
            fusion_hidden_states = torch.cat((user_rag_emb, hidden_states), dim=1)
            user_mask = torch.ones(user_rag_emb.size(0), user_rag_emb.size(1)).to(hidden_states.device)
            fusion_attention_mask = torch.cat((user_mask, attention_mask), dim=1)
        elif user_rag_emb is not None and text_encoder_outputs is not None:
            fusion_hidden_states = torch.cat((text_encoder_outputs, user_rag_emb, hidden_states), dim=1)
            user_mask = torch.ones(user_rag_emb.size(0), user_rag_emb.size(1)).to(hidden_states.device)
            fusion_attention_mask = torch.cat((text_attention_mask, user_mask, attention_mask), dim=1)
        elif user_rag_emb is None and text_encoder_outputs is not None:
            fusion_hidden_states = torch.cat((text_encoder_outputs, hidden_states), dim=1)
            fusion_attention_mask = torch.cat((text_attention_mask, attention_mask), dim=1)
        else:
            fusion_hidden_states = hidden_states
            fusion_attention_mask = attention_mask
        
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            
            decoder_input_ids = self._shift_right(labels)
        if decoder_inputs_embeds is not None:
            batch_size = decoder_inputs_embeds.size(0)
            bos_token_emb = self.shared.weight[self.config.decoder_start_token_id].unsqueeze(0)# [1, embedding_dim]
            bos_token_emb = bos_token_emb.expand(batch_size, -1, -1)  # 变成 [batch_size, 1, embedding_dim]
            decoder_inputs_embeds = torch.cat([bos_token_emb, decoder_inputs_embeds], dim=1)
        
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                fusion_attention_mask = fusion_attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds.to(self.decoder.first_device)
        
        if decoder_inputs_embeds is not None:
            decoder_outputs = self.decoder(
                input_ids=None,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,

                encoder_hidden_states=fusion_hidden_states,
                encoder_attention_mask=fusion_attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,

                encoder_hidden_states=fusion_hidden_states,
                encoder_attention_mask=fusion_attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = decoder_outputs[0]

        
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        
        
        loss = None
        if labels is not None:
            loss = self.total_loss(lm_logits, labels, decoder_input_ids)

        # ------------------------------------------

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        return LETTER_Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=sequence_output,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=fusion_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_mask=fusion_attention_mask,
        )
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=2048, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.Linear(projection_dim, output_dim),
        )

    def forward(self, x):
        return self.projection(x)  
    
class final_model(nn.Module):

    def __init__(self, T5config, args):
        super(final_model, self).__init__()
        print(T5Config)

        dataset = args.dataset
        self.train_mode = args.train_mode
        self.id_model = LETTER(T5config, dataset, args.model_type)
        self.text_model = LETTER(T5config, dataset, args.model_type)
  
        self.text_model.decoder = self.id_model.decoder
        self.text_model.lm_head = self.id_model.lm_head
        self.text_model.shared = self.id_model.shared

        self.text_emb = np.load(f'../data/llama2/{dataset}.emb-two-llama2-td.npy')

        self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        self.train_user_mean_emb = np.load(f"./rag_need/{args.model_type}/{dataset}/train/user_emb_mean.npy")
        self.val_user_mean_emb = np.load(f"./rag_need/{args.model_type}/{dataset}/val/user_emb_mean.npy")
        self.train_text_user_mean_emb = np.load(f"./rag_need/{args.model_type}/{dataset}/train/text_user_emb_mean.npy")
        self.val_text_user_mean_emb = np.load(f"./rag_need/{args.model_type}/{dataset}/val/text_user_emb_mean.npy")

        self.train_sparse_index = json.load(open(f"./rag_need/{args.model_type}/{dataset}/train/rag_user_index.json"))
        self.train_sparse_index = np.array(self.train_sparse_index)
        self.train_sas_index = json.load(open(f"./rag_need/{args.model_type}/{dataset}/train/item_retrival.json"))
        self.train_sas_index = np.array(self.train_sas_index)
        self.val_sparse_index = json.load(open(f"./rag_need/{args.model_type}/{dataset}/val/rag_user_index.json"))
        self.val_sparse_index = np.array(self.val_sparse_index)
        self.val_sas_index = json.load(open(f"./rag_need/{args.model_type}/{dataset}/val/item_retrival.json"))
        self.val_sas_index = np.array(self.val_sas_index)
        self.train_rerank = json.load(open(f"./rag_need/{args.model_type}/{dataset}/train/reranked_user_index.json"))
        self.train_rerank = np.array(self.train_rerank)
        self.val_rerank = json.load(open(f"./rag_need/{args.model_type}/{dataset}/val/reranked_user_index.json"))
        self.val_rerank = np.array(self.val_rerank)
        self.text_emb = torch.from_numpy(self.text_emb).float()
        self.text_emb = torch.cat((self.text_emb, torch.zeros(4096).unsqueeze(0)), dim=0)
        self.projection_head = ProjectionHead(input_dim=4096, projection_dim=2048, output_dim=128)
        
    def kl_contrastive_loss(self, normal_logits, tail_logits, temperature=0.95):
        normal_probs = F.softmax(normal_logits / temperature, dim=-1)  # [B, L, V]
        tail_probs = F.softmax(tail_logits / temperature, dim=-1)     # [B, L, V]

        
        kl_loss = (
            F.kl_div(
                F.log_softmax(tail_logits / temperature, dim=-1),
                normal_probs,
                log_target=False,
                reduction='none'
            ).sum(-1) + 
            F.kl_div(
                F.log_softmax(normal_logits / temperature, dim=-1),
                tail_probs,
                log_target=False,
                reduction='none'
            ).sum(-1)
        )   

        
        return kl_loss.mean()
    def Con_loss(self, rep1, rep2):
        logits = torch.matmul(rep1, rep2.T) / 0.9 
        labels = torch.arange(rep1.size(0), device=rep1.device)
        return F.cross_entropy(logits, labels)

    def get_prob(self, epoch, start_epoch=0, max_epoch=150, max_prob=0.3):
        if epoch < start_epoch:
            return 0
        return max_prob
        
    def forward(self, batch, epoch):
        device = next(self.parameters()).device
        self.text_emb = self.text_emb.to(device)
        
        input_ids = batch['lm_inputs']['input_ids'].to(device)
        attention_mask = batch['lm_inputs']['attention_mask'].to(device)
        labels = batch['lm_inputs']['labels'].to(device)
        index = batch['lm_inputs']['index']
        sim_sparse_user_index = self.train_sparse_index[index][:,:30][:, ::-1]
        sim_sas_user_index = self.train_sas_index[index][:,:30][:, ::-1]
        sim_reranked_user_index = self.train_rerank[index][:,:30]
        input_item_idx = batch['text_input_ids'].to(device)
        input_item_mask = batch['text_attention_mask'].to(device)
        id_user_rag_emb = self.train_user_mean_emb[sim_reranked_user_index]
        id_user_rag_emb = torch.from_numpy(id_user_rag_emb).float().to(device)
        id_inputs = {
                'input_ids': batch['lm_inputs']['input_ids'].to(device),
                'attention_mask': batch['lm_inputs']['attention_mask'].to(device),
                'labels': batch['lm_inputs']['labels'].to(device),
                'user_rag_emb':  id_user_rag_emb ,
                'output_hidden_states': True
            }

        id_model_output = self.id_model(**id_inputs)
        return id_model_output.loss 

            

    def evaluate(self, batch, index):
        device = next(self.parameters()).device
        
        self.text_emb = self.text_emb.to(device)
        
        input_ids = batch['lm_inputs']['input_ids'].to(device)
        attention_mask = batch['lm_inputs']['attention_mask'].to(device)
        labels = batch['lm_inputs']['labels'].to(device)
        index = batch['lm_inputs']['index']
        sim_sparse_user_index = self.val_sparse_index[index][:,:30][:, ::-1]
        sim_sas_user_index = self.val_sas_index[index][:,:30][:, ::-1]
        sim_reranked_user_index = self.val_rerank[index][:,:30]
        input_item_idx = batch['text_input_ids'].to(device)
        input_item_mask = batch['text_attention_mask'].to(device)
        id_user_rag_emb = self.val_user_mean_emb[sim_reranked_user_index]
        id_user_rag_emb = torch.from_numpy(id_user_rag_emb).float().to(device)

            
        id_inputs = {   
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states':True,
            'labels': labels,
            'user_rag_emb': id_user_rag_emb,
            'return_dict': True,
        }

        
        id_encoder_output = self.id_model(**id_inputs)
        return  id_encoder_output.loss 
    
    def predict(self, batch, prefix_allowed_tokens):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            input = batch[0].to(device)
            self.text_emb = self.text_emb.to(device)
        
            input_ids = batch[0]['input_ids'].to(device)
            attention_mask = batch[0]['attention_mask'].to(device)
            index = batch[1]
            id_inputs = {   
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            input_item_idx = batch[3].to(device)
            input_item_mask = batch[4].to(device)

            text_inputs = {
                'inputs_embeds': self.projection_head(self.text_emb[input_item_idx]),
                'attention_mask': input_item_mask,
            }
            text_model_output = self.text_model.encoder(**text_inputs)

            text_model_output = text_model_output.last_hidden_state
            sim_sparse_user_index = self.id_model.test_sparse_index[index][:,:30][:, ::-1]
            sim_sas_user_index = self.id_model.test_sas_index[index][:,:30][:, ::-1]
            sim_reranked_user_index = self.id_model.test_rerank[index][:,:30]
            id_user_rag_emb = self.id_model.user_mean_emb[sim_reranked_user_index]
            id_user_rag_emb = torch.from_numpy(id_user_rag_emb).float().to(device)

            if self.train_mode == 'rag':
                output = self.id_model.generate(input_ids = input_ids, attention_mask = attention_mask,
                                                max_new_tokens=10,
                                                # max_length=10,
                                                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                                                num_beams=20,
                                                num_return_sequences=20,
                                                output_scores=True,
                                                return_dict_in_generate=True,
                                                early_stopping=True,
                                                user_rag_emb = id_user_rag_emb,)
            else:
                '''
                text_encoder_outputs=None, 
                text_attention_mask=None,
                user_rag_emb = None,
                '''
                output = self.id_model.generate(input_ids = input_ids, attention_mask = attention_mask,
                                                max_new_tokens=10,
                                                # max_length=10,
                                                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                                                num_beams=20,
                                                num_return_sequences=20,
                                                output_scores=True,
                                                return_dict_in_generate=True,
                                                early_stopping=True,
                                                text_encoder_outputs=text_model_output,
                                                text_attention_mask=input_item_mask,
                                                user_rag_emb=id_user_rag_emb,

                                                )
        return output


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
import random
from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer

import math
from math import sqrt


class LETTER(T5ForConditionalGeneration):

    def __init__(self, config: T5Config):

        super().__init__(config)

        
        self.temperature = 1.0

    def set_hyper(self,temperature):
        self.temperature = temperature


    def ranking_loss(self, lm_logits, labels):
        if labels is not None:
            t_logits = lm_logits/self.temperature
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            
            labels = labels.to(lm_logits.device)
            loss = loss_fct(t_logits.view(-1, t_logits.size(-1)), labels.view(-1))
        return loss


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
        if text_encoder_outputs is not None and text_attention_mask is not None:
            hidden_states = torch.cat((text_encoder_outputs, hidden_states), dim=1)
            attention_mask = torch.cat((text_attention_mask, attention_mask), dim=1)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            
            decoder_input_ids = self._shift_right(labels)
            
        if decoder_inputs_embeds is not None:
            batch_size = decoder_inputs_embeds.size(0)
            bos_token_emb = self.shared.weight[self.config.decoder_start_token_id].unsqueeze(0)
            bos_token_emb = bos_token_emb.expand(batch_size, -1, -1)  
            decoder_inputs_embeds = torch.cat([bos_token_emb, decoder_inputs_embeds], dim=1)
            
        
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
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

                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
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

                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
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
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        loss = None
        
        if labels is not None:
            
            loss = self.total_loss(lm_logits, labels, decoder_input_ids)

        
        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
import faiss

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
        self.args = args
        self.id_model = LETTER(T5config)
        self.text_model = LETTER(T5config)
        self.dataset_name = self.args.dataset
        self.text_model.decoder = self.id_model.decoder
        self.text_model.lm_head = self.id_model.lm_head
        self.text_model.shared = self.id_model.shared
        text_emb = np.load(f'../data/{self.dataset_name}.emb-two-llama2-td.npy')
        image_emb = np.load(f'../data/{self.dataset_name}.emb-ViT-L-14.npy')
        self.text_emb = torch.tensor(text_emb, dtype=torch.float32)
        self.padding_emb = torch.zeros(1, 4096, dtype=torch.float32)
        self.text_emb = torch.cat([self.text_emb, self.padding_emb], dim=0)
        self.image_emb = torch.tensor(image_emb, dtype=torch.float32)
        self.padding_emb = torch.zeros(1, 768, dtype=torch.float32)
        self.image_emb = torch.cat([self.image_emb, self.padding_emb], dim=0)
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        self.projection_head = ProjectionHead(input_dim=4096, projection_dim=2048, output_dim=128)
        self.image_projection_head = ProjectionHead(input_dim=768,projection_dim=512, output_dim=128)
    def kl_contrastive_loss(self, normal_logits, tail_logits, temperature=0.85):
        
        normal_probs = F.softmax(normal_logits / temperature, dim=-1)  
        tail_probs = F.softmax(tail_logits / temperature, dim=-1)     
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
    def Con_loss(self, rep1, rep2, mask=None):
        logits = torch.matmul(rep1, rep2.T) / 0.9 
        labels = torch.arange(rep1.size(0), device=rep1.device)
        return F.cross_entropy(logits, labels)
    def get_prob(self, epoch, start_epoch=0, max_epoch=150, max_prob=0.3):
        if epoch < start_epoch:
            return 0
        return max_prob
        
        return min(max_prob, (epoch - start_epoch) / (max_epoch - start_epoch) * max_prob)

    def compute_mean(self, output, mask):
        masked_output = output * mask.unsqueeze(-1)
        return masked_output.sum(1) / mask.sum(1, keepdim=True)
    def item_con_loss(self, normal, tail):
        assert normal.size(1) == 4 * tail.size(1)
        normal = normal.reshape(normal.size(0), tail.size(1), -1, normal.size(-1)).sum(dim=2)
        normal = normal.reshape(-1, normal.size(-1))
        tail = tail.reshape(-1, tail.size(-1))
        loss_1 = self.Con_loss(normal, tail)
        loss_2 = self.Con_loss(tail, normal)
        return loss_1 + loss_2
    def forward(self, batch, epoch):
        device = next(self.parameters()).device
        self.text_emb = self.text_emb.to(device)
        self.image_emb = self.image_emb.to(device)
        input_image_emb = self.image_projection_head(self.image_emb[batch['text_input_ids'].to(device)])
        input_text_emb = self.projection_head(self.text_emb[batch['text_input_ids'].to(device)])
        id_type = torch.tensor(batch['id_type'], dtype=torch.long, device=device)
        combine_emb = torch.where(id_type.unsqueeze(-1).unsqueeze(-1) == 0, input_text_emb, input_image_emb)
        prompt_emb = self.id_model.encoder.get_input_embeddings()(batch['lm_inputs']['input_ids'][:,:4].to(device))
        combine_emb = torch.cat([prompt_emb, combine_emb], dim=1)
        combine_mask = torch.ones(combine_emb.size(0), 4, dtype=torch.bool).to(device)
        combine_mask = torch.cat([combine_mask, batch['text_attention_mask'].to(device)], dim=1)
        text_inputs = {
            'inputs_embeds': combine_emb,
            'attention_mask': combine_mask,
            'labels': batch['lm_inputs']['labels'].to(device),
            'output_hidden_states': True
        }
        text_model_output = self.text_model(**text_inputs)
        
        if random.random() < self.get_prob(epoch, start_epoch=10, max_epoch=150, max_prob=0.0):
            
            text_logits = text_model_output.logits[:,:-1,:]
            k = 5
            emb_weights, top_k_tgt = text_logits.topk(k, dim=-1)

            
            top_k_tgt = top_k_tgt.unsqueeze(-2)  
            k_embs = self.id_model.decoder.get_input_embeddings()(top_k_tgt).transpose(2, 3)
            
            emb_weights /= (emb_weights.sum(dim=-1, keepdim=True) + 1e-9)  

            
            weights = emb_weights.unsqueeze(3)  
            
            k_embs = k_embs.squeeze(3).transpose(2, 3)  
            weights = emb_weights.unsqueeze(-1)  

            batch_size, seq_len, emb_size, k = k_embs.shape
            model_prediction_emb = torch.bmm(k_embs.view(-1, emb_size, k), weights.view(-1, k, 1))
            model_prediction_emb = model_prediction_emb.view(batch_size, seq_len, emb_size)
            

            gold_embs = self.id_model.decoder.get_input_embeddings()(batch['lm_inputs']['labels'][:,:-1].to(device))
            for i in range(1, gold_embs.size(1)):
                if  random.random() < self.get_prob(epoch, start_epoch=10, max_epoch=150, max_prob=0.4):
                    gold_embs[:,i,:] = model_prediction_emb[:,i,:]
            gold_embs = gold_embs.to(device)
            id_inputs = {
                'input_ids': batch['lm_inputs']['input_ids'].to(device),
                'attention_mask': batch['lm_inputs']['attention_mask'].to(device),
                'labels': batch['lm_inputs']['labels'].to(device),
                'decoder_inputs_embeds': gold_embs,
                'output_hidden_states': True
            }
        else:
            id_inputs = {
                'input_ids': batch['lm_inputs']['input_ids'].to(device),
                'attention_mask': batch['lm_inputs']['attention_mask'].to(device),
                'labels': batch['lm_inputs']['labels'].to(device),
                'output_hidden_states': True
            }
        id_model_output = self.id_model(**id_inputs)
        normal_mean = self.compute_mean(id_model_output.encoder_last_hidden_state, id_inputs['attention_mask'])
        tail_mean = self.compute_mean(text_model_output.encoder_last_hidden_state, text_inputs['attention_mask'])
        
        normal = id_model_output.encoder_last_hidden_state * id_inputs['attention_mask'].unsqueeze(-1)
        tail = text_model_output.encoder_last_hidden_state * text_inputs['attention_mask'].unsqueeze(-1)
        normal = normal[:,4:-1,:]
        tail = tail[:,4:-1,:]
        item_loss = self.item_con_loss(normal, tail)
        normal_logits = id_model_output.logits[:,:-1,:]
        tail_logits = text_model_output.logits[:,:-1,:]
        logit_loss = self.kl_contrastive_loss(normal_logits, tail_logits)
        con_loss_1 = self.Con_loss(normal_mean, tail_mean)
        con_loss_2 = self.Con_loss(tail_mean, normal_mean)
        con_loss = con_loss_1 + con_loss_2
        tail_loss = text_model_output.loss
        normal_loss = id_model_output.loss
        
        return normal_loss + tail_loss  +  0.5 * logit_loss + 0.1 * item_loss + 0.5 * con_loss, normal_mean, tail_mean
        

    def evaluate(self, batch, index, train_mode='train', data_mode='eval'):
        device = next(self.parameters()).device
        
        self.text_emb = self.text_emb.to(device)
        input_ids = batch['lm_inputs']['input_ids'].to(device)
        attention_mask = batch['lm_inputs']['attention_mask'].to(device)
        labels = batch['lm_inputs']['labels'].to(device)
        id_inputs = {   
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'output_hidden_states':True
        }
        
        id_model_output = self.id_model(**id_inputs)
        return id_model_output.loss 

    def predict(self, batch, prefix_allowed_tokens, test_task='seqrec', test_mode='test', data_mode='test', index=None):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            self.text_emb = self.text_emb.to(device)
            input_ids = batch[0]['input_ids'].to(device)
            attention_mask = batch[0]['attention_mask'].to(device)
            input_item_idx = batch[2].to(device)
            input_item_mask = batch[3].to(device)
            text_inputs = {
            'inputs_embeds': self.projection_head(self.text_emb[input_item_idx]),
            'attention_mask': input_item_mask,
            }
            if test_mode == 'test':
                output = self.id_model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                max_new_tokens=10,
                                                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                                                num_beams=20,
                                                num_return_sequences=20,
                                                output_scores=True,
                                                return_dict_in_generate=True,
                                                early_stopping=True,
                                                )
            elif test_mode == 'rag':
                
                text_model_output = self.text_model.encoder(**text_inputs)
                text_model_output = text_model_output.last_hidden_state
                output = self.id_model.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                max_new_tokens=10,
                                                prefix_allowed_tokens_fn=prefix_allowed_tokens,
                                                num_beams=1,
                                                num_return_sequences=1,
                                                output_scores=True,
                                                return_dict_in_generate=True,
                                                early_stopping=True,
                                                output_hidden_states=True,
                                                )
                encoder_output = output.encoder_hidden_states[-1]
                encoder_output = encoder_output * attention_mask.unsqueeze(-1)
                encoder_output_mean = encoder_output.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
                text_model_output = text_model_output * input_item_mask.unsqueeze(-1)
                text_model_output_mean = text_model_output.sum(dim=1) / input_item_mask.sum(dim=1).unsqueeze(-1)
                decoder_output = output.decoder_hidden_states[-1]
                temp_decoder_output = decoder_output[0]
                for i in range(1, len(decoder_output) - 1):
                    temp_decoder_output = torch.cat((temp_decoder_output, decoder_output[i]), dim=1)
                decoder_output_mean = temp_decoder_output.mean(dim=1)
                np.save(f'../MQL4Rec/mutil/{self.dataset_name}/{test_task}/{data_mode}/{index}.mean.npy', encoder_output_mean.cpu().detach().numpy())
                np.save(f'../MQL4Rec/mutil/{self.dataset_name}/{test_task}/{data_mode}/{index}.decoder.mean.npy', decoder_output_mean.cpu().detach().numpy())
                np.save(f'../MQL4Rec/mutil/{self.dataset_name}/{test_task}/{data_mode}/{index}.text.mean.npy', text_model_output_mean.cpu().detach().numpy())
        return output


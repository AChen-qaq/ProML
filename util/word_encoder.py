import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.utils.rnn import pad_sequence



class BERTWordEncoder(nn.Module):

    def __init__(self, pretrain_path, tokenizer = None): 
        nn.Module.__init__(self)
        self.bert = nn.DataParallel(BertModel.from_pretrained(pretrain_path))
        self.tokenizer = tokenizer

    def forward(self, words, masks):
        outputs = self.bert(words, attention_mask=masks, output_hidden_states=True, return_dict=True)
        #outputs = self.bert(inputs['word'], attention_mask=inputs['mask'], output_hidden_states=True, return_dict=True)
        # use the sum of the last 4 layers
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        
        word_embeddings = torch.mean(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768]
        del outputs, last_four_hidden_states
        return word_embeddings

    def encode_multisupport_singlequery(self, support_sents, support_masks, query_sent, query_mask):
        
        def merge(support_sents, support_masks, query_sent, query_mask):
            support_text = torch.cat([seq[mask == 1] for seq, mask in zip(support_sents, support_masks)])
            query_text = query_sent[query_mask == 1]
            input = torch.cat([torch.tensor([self.tokenizer.cls_token_id], device = 'cuda'), query_text, torch.tensor([self.tokenizer.sep_token_id], device = 'cuda'), support_text, torch.tensor([self.tokenizer.sep_token_id], device = 'cuda')])
            del support_text, query_text
            return input
        inputs = [merge(support_sents[i: i+5], support_masks[i: i+5], query_sent, query_mask) for i in range(0, len(support_sents), 5)]
        lens = [len(x) for x in inputs]
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        mask = padded_inputs != self.tokenizer.pad_token_id
        del inputs
        # print(padded_inputs)
        # print(mask)
        return padded_inputs, mask, lens

    def encode_multisupport_singlequery_tanl(self, support_sents, support_masks, query_sent, query_mask):
        
        def merge(support_sents, support_masks, query_sent, query_mask):
            support_text = torch.cat(support_sents, dim=0)
            query_text = query_sent[query_mask == 1]
            inputs = torch.cat([torch.tensor([self.tokenizer.cls_token_id], device = 'cuda'), query_text, torch.tensor([self.tokenizer.sep_token_id], device = 'cuda'), support_text, torch.tensor([self.tokenizer.sep_token_id], device = 'cuda')])
            
            text_mask = torch.cat(support_masks, dim=0)
            
            text_mask = torch.cat([torch.zeros(len(query_text)+2, device = 'cuda'), text_mask, torch.tensor([0], device = 'cuda')], dim=0)
            token_types = torch.cat([torch.zeros(len(query_text)+2, device = 'cuda'), torch.ones(len(support_text)+1, device = 'cuda')], dim=0)
            del support_text, query_text
            return inputs, text_mask, token_types
        inputs = []
        support_text_mask = []
        token_types = []
        for i in range(0, len(support_sents), 5):
            tmp = merge(support_sents[i: i+5], support_masks[i: i+5], query_sent, query_mask)
            inputs.append(tmp[0])
            support_text_mask.append(tmp[1])
            token_types.append(tmp[2])
        # inputs = [merge(support_sents[i: i+5], support_masks[i: i+5], query_sent, query_mask) for i in range(0, len(support_sents), 5)]
        
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_support_text_mask = pad_sequence(support_text_mask, batch_first=True, padding_value=0)
        padded_token_types = pad_sequence(token_types, batch_first=True, padding_value=0)
        mask = padded_inputs != self.tokenizer.pad_token_id

        
        # print(padded_inputs)
        # print(mask)

        outputs = self.bert(padded_inputs, attention_mask=mask, token_type_ids=padded_token_types.long(), output_hidden_states=True, return_dict=True)
        #outputs = self.bert(inputs['word'], attention_mask=inputs['mask'], output_hidden_states=True, return_dict=True)
        # use the sum of the last 4 layers
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        
        word_embeddings = torch.mean(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768]

        del inputs, outputs, last_four_hidden_states, padded_inputs, mask

        return word_embeddings, padded_support_text_mask
    
    def encode_list(self, tokens, masks):
        padded_inputs = pad_sequence([x for x in tokens], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        padded_support_text_mask = pad_sequence([x for x in masks], batch_first=True, padding_value=0)
        mask = padded_inputs != self.tokenizer.pad_token_id
        outputs = self.bert(padded_inputs, attention_mask=mask, output_hidden_states=True, return_dict=True)
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        
        word_embeddings = torch.mean(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768]
        embs = word_embeddings[padded_support_text_mask == 1]
        del padded_inputs, padded_support_text_mask, mask, outputs, last_four_hidden_states, word_embeddings
        return embs.view(-1, embs.size(-1))


    def construct_batch_data(self, support_sents, support_masks, query_sents, query_masks):
        bsz = len(query_sents)
        support_text = torch.cat([seq[mask == 1] for seq, mask in zip(support_sents, support_masks)])
        query_texts = [query_sents[i][query_masks[i] == 1] for i in range(bsz)]
        inputs = [torch.cat([torch.tensor([self.tokenizer.cls_token_id], device = 'cuda'), query_texts[i], torch.tensor([self.tokenizer.sep_token_id], device = 'cuda'), support_text, torch.tensor([self.tokenizer.sep_token_id], device = 'cuda')]) for i in range(bsz)]
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=-1)
        mask = padded_inputs != -1
        return inputs, mask



    def prompt_encode(self, support_sents, support_masks, query_sent, query_mask):
        support_text = torch.cat([seq[mask == 1] for seq, mask in zip(support_sents, support_masks)])
        query_text = query_sent[query_mask == 1]
        inputs = torch.cat([torch.tensor([self.tokenizer.cls_token_id], device = 'cuda'), query_text, torch.tensor([self.tokenizer.sep_token_id], device = 'cuda'), support_text, torch.tensor([self.tokenizer.sep_token_id], device = 'cuda')])
        outputs = self.bert(inputs.unsqueeze(0), token_type_ids=torch.tensor([0]*(2+len(query_text))+[1]*(1+len(support_text)), device = 'cuda').unsqueeze(0), output_hidden_states=True, return_dict=True)
        last_four_hidden_states = torch.cat([hidden_state.unsqueeze(0) for hidden_state in outputs['hidden_states'][-4:]], 0)
        
        word_embeddings = torch.mean(last_four_hidden_states, 0) # [num_sent, number_of_tokens, 768]
        del outputs, inputs, support_text, query_text, last_four_hidden_states
        return word_embeddings.squeeze(0)

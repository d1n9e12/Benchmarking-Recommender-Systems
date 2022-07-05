# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:47:19 2021

@author: Daniel
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import pickle

#from datetime import date
import random

import torch
import torch.backends.cudnn as cudnn
from torch import optim as optim
from abc import *

import torch.utils.data as data_utils
from scipy import sparse
import numpy as np
#import math, changed match. operations to np.

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from abc import ABCMeta, abstractmethod


from tqdm import trange

import ray    
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune import CLIReporter
from functools import partial



#######################################################################
#ADD ones to each item because the example from the amazon reviews will be used
def BertTrainDataset(u2seq, max_len, mask_prob, mask_token, num_items, rng):
    """
    u2seq = train
    users = sorted(u2seq.keys())
    max_len = max_len
    mask_prob = mask_prob
    mask_token = CLOZE_MASK_TOKEN 
    num_items = num_items
    rng = rng
    """
    users = sorted(u2seq.keys())
    tokens_l, labels_l = [], []
    for user in users:
        seq = u2seq[user]
        tokens = []
        labels = []
        for s in seq:
            prob = rng.random()
            if prob < mask_prob:
                prob /= mask_prob
    
                if prob < 0.8:
                    tokens.append(mask_token)
                elif prob < 0.9:
                    tokens.append(rng.randint(1, num_items))
                else:
                    tokens.append(s)
    
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)
    
        tokens = tokens[-max_len:]
        labels = labels[-max_len:]
    
        mask_len = max_len - len(tokens)
    
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        tokens_l.append(tokens)
        labels_l.append(labels)
    
    return torch.LongTensor(tokens_l), torch.LongTensor(labels_l)

def BertEvalDataset(u2seq, u2answer, max_len, mask_token, negative_samples):
    """
    u2seq = train
    users = sorted(u2seq.keys())
    u2answer = val
    max_len = max_len
    mask_token = CLOZE_MASK_TOKEN 
    negative_samples = train_negative_samples
    """
    users = sorted(u2seq.keys())
    seq_l, cand_l, labels_l = [], [], []
    for index in range(len(users)):
        user = users[index]
        seq = u2seq[user]
        answer = u2answer[user]
        negs = negative_samples[user]
    
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
    
        seq = seq + [mask_token]
        seq = seq[-max_len:]
        padding_len = max_len - len(seq)
        seq = [0] * padding_len + seq
        
        seq_l.append(seq)
        cand_l.append(candidates)
        labels_l.append(labels)
    
    return torch.LongTensor(seq_l), torch.LongTensor(cand_l), torch.LongTensor(labels_l) 




#########################################################################################
#MODEL
#########################################################################################

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    @abstractmethod
    def code(cls):
        pass

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        # self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)  # + self.segment(segment_label)
        return self.dropout(x)
    
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / np.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)   
    


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)    
    

class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()

        #fix_random_seed_as(config["SEED"])
        # self.init_weights()

        max_len = config["max_len"]
        num_items = config["n_items"]
        n_layers = config["num_blocks"]
        heads = config["num_heads"]
        vocab_size = num_items + 2
        hidden = config["hidden_units"]
        self.hidden = hidden
        dropout = config["dropout_rate"]

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass    

class BERTModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BERT(config)
        self.out = nn.Linear(self.bert.hidden, config["n_items"] + 1)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)


def train_bert(config, checkpoint_dir = None):
    #fix_random_seed_as(1)
    model = BERTModel(config)
    device = "cpu"    
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    #optimizer = optim.Adam(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr = config["lr"], weight_decay = config["wd"])

    train_loss = 0
    for epoch in range(config["epochs"]):
        model.train()
        train_dataset = BertTrainDataset(train_seq, config["max_len"], config["mask_prob"],
                                         CLOZE_MASK_TOKEN, num_items, rng)
        #for batch in minibatch(tokens_train, labels_train, config["batch_size"]): #when passed through minibatch the number of dims disappears to one
        for batch_idx, start_idx in enumerate(range(0, len(umap), config["batch_size"])): #batchsize
            end_idx = min(start_idx + config["batch_size"], len(umap)) #batch_size
            #batch = tokens_train[start_idx: end_idx], labels_train[start_idx: end_idx]
            batch = train_dataset[0][start_idx: end_idx], train_dataset[1][start_idx: end_idx]
            batch_size = batch[0].size(0)
            batch = [x.to(device) for x in batch]
    
            optimizer.zero_grad()
            seqs, labels = batch
            logits = model(seqs)  # B x T x V
            logits = logits.view(-1, logits.size(-1))  # (B*T) x V
            labels = labels.view(-1)  # B*T
            loss = nn.CrossEntropyLoss(ignore_index=0)(logits, labels)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print(loss)
            
        
        model.eval()
        val_dataset = BertEvalDataset(train_seq, val_seq, config["max_len"],
                                      CLOZE_MASK_TOKEN, train_negative_samples)    
        with torch.no_grad():
            prec100_l, rec100_l, f100_l, hit100_l = [], [], [], []
            ndcg100_l, mrr100_l, arhr100_l, map100_l = [], [], [], []
            div100_l, nov100_l, ser100_l, cc100_l, dc100_l = [], [], [], [], []
            
            for batch_idx, start_idx in enumerate(range(0, len(umap), config["batch_size"])): #batchsize
                end_idx = min(start_idx + config["batch_size"], len(umap)) #batch_size
                #batch = seq_val[start_idx: end_idx], candidates_val[start_idx: end_idx], labels_val[start_idx: end_idx]
                batch = val_dataset[0][start_idx: end_idx], val_dataset[1][start_idx: end_idx], val_dataset[2][start_idx: end_idx]
                batch = [x.to(device) for x in batch]
                seqs, candidates, labels = batch
                scores = model(seqs)  # B x T x V
                
                scores = scores[:, -1, :]  # B x V
                scores = scores.gather(1, candidates)  # B x C
                prec100_l.append(precision_sequence(scores, labels, 100))
                rec100_l.append(recall_sequence(scores, labels, 100))
                f100_l.append(f1_sequence(scores, labels, 100))
                hit100_l.append(hits_sequence(scores, labels, 100))
                ndcg100_l.append(ndcg_sequence(scores, labels, 100))
                mrr100_l.append(mrr_sequence(scores, labels, 100))
                arhr100_l.append(arhr_sequence(scores, labels, 100))
                map100_l.append(map_sequence(scores, labels, 100))
                
                total_train = torch.cat((train_dataset[0][start_idx: end_idx], val_dataset[0][start_idx: end_idx]), 1)
                beyond = Beyond_accuracy_prep_bert(scores, total_train , 20)
                div100_l.append(diversity(beyond[0], beyond[1]))
                nov100_l.append(novelty(beyond[0], beyond[1]))
                ser100_l.append(serendipity(beyond[0], beyond[1]))
                cc100_l.append(catalog_coverage(beyond[0], beyond[1]))
                dc100_l.append(distributional_coverage(beyond[0], beyond[1]))
        
            prec_score = np.mean(prec100_l)    
            rec_score = np.mean(rec100_l)
            f1_score = np.mean(f100_l)
            hit_score = np.mean(hit100_l)
            ndcg_score = np.mean(ndcg100_l)
            mrr_score = np.mean(mrr100_l)
            arhr_score = np.mean(arhr100_l)
            map_score = np.mean(map100_l)
            div_score = np.mean(div100_l)
            nov_score = np.mean(nov100_l)
            ser_score = np.mean(ser100_l)
            cc_score = np.mean(cc100_l)
            dc_score = np.mean(dc100_l)
            
        
            tune.report(prec100 = prec_score, rec100 = rec_score, f100 = f1_score, hit100 = hit_score, 
                        ndcg100 = ndcg_score, mrr100 = mrr_score, arhr100 = arhr_score, map100 = map_score,
                        div100 = div_score, nov100 = nov_score, ser100 = ser_score, cc100 = cc_score, dc100 = dc_score)


config = {
    "batch_size": 20,
    "lr": 1e-3, "wd" : 1e-3,
    "max_len": 50,#should be the max len seq there is
    "hidden_units": 256,
    "num_blocks": 2, 
    "epochs": 10,
    "num_heads": 4,
    "mask_prob": 0.15,
    "dropout_rate": 0.1,
    "gamma": 1,
    "decay_step": 25,
    "enable_lr_schedule": True,
    #"SEED": 1,
    "device_idx": "0",
    "num_gpu": 0,
    "n_items": n_items
    }
    
config = {
    "batch_size": tune.choice([1, 2, 4, 8, 16]),
    "lr": tune.choice([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    "wd": tune.choice([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    "max_len": tune.choice(list(range(100))),#should be the max len seq there is
    "hidden_units": 256,
    "num_blocks": 2, 
    "epochs": 10,
    "num_heads": 4,
    "mask_prob": 0.15,
    "dropout_rate": 0.1,
    "gamma": 1,
    "decay_step": 25,
    "enable_lr_schedule": True,
    #"SEED": 1,
    "device_idx": "0",
    "num_gpu": 0,
    }

config["n_items"] = len(smap)
#MULTIPLE OPTIMIZERS HERE SGD AND ADAm

       
scheduler = ASHAScheduler(
        metric= "ndcg100",
        mode="max",) 
 
reporter = CLIReporter(metric_columns=["prec100", "rec100", "f100", "hit100",
                                       "ndcg100", "mrr100", "arhr100", "map100", "mar100",
                                       "div100", "nov100", "ser100", "cc100", "dc100"])
                                       
#bayesopt = BayesOptSearch(metric="ndcg", mode="max") #wait, max or min???????

result_bert = tune.run(
    partial(train_bert),
    #resources_per_trial={"cpu": 8}, #, "gpu": gpus_per_trial},
    config=config,
    num_samples= 10,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end= False,
    fail_fast = "raise",
    #search_alg = bayesopt
    )


best_trial_bert = result_sasrec.get_best_trial("ndcg100", "max", "last")
print("Best trial config: {}".format(best_trial_bert.config))

print("Best trial final validation precision: {}".format(best_trial_bert.last_result["prec100"]))
print("Best trial final validation recall: {}".format(best_trial_bert.last_result["rec100"]))
print("Best trial final validation f1: {}".format(best_trial_bert.last_result["f100"]))
print("Best trial final validation hits: {}".format(best_trial_bert.last_result["hit100"]))

print("Best trial final validation ndcg: {}".format(best_trial_bert.last_result["ndcg100"]))
print("Best trial final validation mrr: {}".format(best_trial_bert.last_result["mrr100"]))
print("Best trial final validation arhr: {}".format(best_trial_bert.last_result["arhr100"]))
print("Best trial final validation MAP: {}".format(best_trial_bert.last_result["map100"]))


print("Best trial final validation novelty: {}".format(best_trial_bert.last_result["nov100"]))
print("Best trial final validation diversity: {}".format(best_trial_bert.last_result["div100"]))
print("Best trial final validation serendipity: {}".format(best_trial_bert.last_result["ser100"]))
print("Best trial final validation catalog coverage: {}".format(best_trial_bert.last_result["cc100"]))
print("Best trial final validation distributional coverage: {}".format(best_trial_bert.last_result["dc100"]))

#fix_random_seed_as(1)
model = BERTModel(config)
device = "cpu"    
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = best_trial_bert.config["lr"], weight_decay = best_trial_bert.config["wd"])

train_loss = 0
for epoch in range(best_trial_bert.last_result["training_iteration"]):
    model.train()
    train_dataset = BertTrainDataset(train_seq, best_trial_bert.config["max_len"], best_trial_bert.config["mask_prob"],
                                     CLOZE_MASK_TOKEN, num_items, rng)
    for batch_idx, start_idx in enumerate(range(0, len(umap), best_trial_bert.config["batch_size"])): #batchsize
        end_idx = min(start_idx + best_trial_bert.config["batch_size"], len(umap)) #batch_size
        #batch = tokens_train[start_idx: end_idx], labels_train[start_idx: end_idx]
        batch = train_dataset[0][start_idx: end_idx], train_dataset[1][start_idx: end_idx]
        batch_size = batch[0].size(0)
        batch = [x.to(device) for x in batch]

        optimizer.zero_grad()
        seqs, labels = batch
        logits = model(seqs)  # B x T x V
        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        labels = labels.view(-1)  # B*T
        loss = nn.CrossEntropyLoss(ignore_index=0)(logits, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print(loss)
        
    
    model.eval()
    val_dataset = BertEvalDataset(train_seq, val_seq, best_trial_bert.config["max_len"],
                                  CLOZE_MASK_TOKEN, train_negative_samples)    
    with torch.no_grad():
        metrics_n = ["PREC@%d", "REC@%d", "F1@%d", "HIT@%d", "NDCG@%d", "MRR@%d", "ARHR@%d", "MAP@%d", "MAR@%d",
             "DIV@%d", "NOV@%d", "SER@%d", "CC@%d", "DC@%d"]
        ks = [1, 5, 10, 20, 50, 100]
        metrics = {metric_n % k: [] for metric_n in metrics_n for k in ks}

        for batch_idx, start_idx in enumerate(range(0, len(umap), best_trial_bert.config["batch_size"])): #batchsize
            end_idx = min(start_idx + best_trial_bert.config["batch_size"], len(umap)) #batch_size
            #batch = seq_val[start_idx: end_idx], candidates_val[start_idx: end_idx], labels_val[start_idx: end_idx]
            batch = val_dataset[0][start_idx: end_idx], val_dataset[1][start_idx: end_idx], val_dataset[2][start_idx: end_idx]
            batch = [x.to(device) for x in batch]
            seqs, candidates, labels = batch
            scores = model(seqs)  # B x T x V
            
            scores = scores[:, -1, :]  # B x V
            scores = scores.gather(1, candidates)  # B x C
            scores = scores.cpu()
            labels = labels.cpu()
            answer_count = labels.sum(1)
            answer_count_float = answer_count.float()
            labels_float = labels.float()
            ranks = (-scores).argsort(dim=1)
            cut = ranks
            for k in sorted(ks, reverse=True):
                cut = cut[:, :k]
                hits = labels_float.gather(1, cut)
                metrics['PREC@%d' % k].append( (hits.sum(1) / torch.LongTensor([k])).mean().item())
                metrics['REC@%d' % k].append((hits.sum(1).float() / torch.min(torch.Tensor([k]), labels.sum(1).float())).mean().cpu().item())
                metrics['F1@%d' % k].append(2 * metrics['PREC@%d' % k] * metrics['REC@%d' % k] / (metrics['PREC@%d' % k] + metrics['REC@%d' % k]))
                metrics['HIT@%d' % k].append(np.mean(np.any(hits.numpy(), axis = 1)))
                
                position = torch.arange(2, 2 + k)
                weights = 1 / torch.log2(position.float())
                dcg = (hits * weights).sum(1)
                idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
                ndcg = (dcg / idcg).mean().item()
                metrics['NDCG@%d' % k].append(ndcg)
        
                rank_l = []
                for user in range(scores.shape[0]):
                    ranks = np.arange(1, hits.shape[1] + 1)[hits.numpy().astype(bool)[0]] #that second dimension had to go
                    if ranks.shape[0] > 0:
                        rank_l.append(1. / ranks[0])
                    else:
                        rank_l.append(0.)  
                metrics['MRR@%d' % k].append(np.mean(rank_l))
        
                position_arhr = torch.arange(1, k + 1)
                weights_arhr = 1 / position_arhr.float()
                arhr = (hits * weights_arhr).sum(1)
        
                metrics['ARHR@%d' % k].append(arhr.mean().item())
        
                p_at_k = hits * torch.cumsum(hits, dim = 1) / (1 + torch.arange(k)) 
                a_p = torch.sum(p_at_k, dim = 1) / torch.min(torch.Tensor([k]), labels.sum(1).float())
                metrics['MAP@%d' % k].append(a_p.mean().item())
                
                a_r = torch.sum(p_at_k, dim = 1) / labels.sum(1).float()  #batch_size results in 1, doesn't seem useful, take shape[1] because it's about the length of the list, but here it's a 2dim array
                metrics['MAR@%d' % k].append(a_r.mean().item())
                                        
                beyond = Beyond_accuracy_prep_bert(recon_batch.detach().numpy(), train_data[e_idxlist[start_idx: end_idx]] , k) #candidates_test[user].numpy()
                metrics['DIV@%d' % k].append(diversity(beyond[0], beyond[1]))
                metrics['NOV@%d' % k].append(novelty(beyond[0], beyond[1]))
                metrics['SER@%d' % k].append(serendipity(beyond[0], beyond[1]))
                metrics['CC@%d' % k].append(catalog_coverage(beyond[0], beyond[1]))
                metrics['DC@%d' % k].append(distributional_coverage(beyond[0], beyond[1]))
                    
    
        

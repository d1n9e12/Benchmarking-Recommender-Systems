# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:11:50 2021

@author: Daniel
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from abc import *
import os
import random
import torch.backends.cudnn as cudnn
from abc import ABCMeta, abstractmethod
from tqdm import trange
import copy
import torch
import random
import time
import ray    
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune import CLIReporter
from functools import partial
from ray.tune.suggest.skopt import SkOptSearch

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


#def sample_function(user_train, usernum, itemnum, maxlen):
def sample(train, user,  n_users, n_items, maxlen):
    seq = np.zeros([maxlen], dtype=np.int32)
    pos = np.zeros([maxlen], dtype=np.int32)
    neg = np.zeros([maxlen], dtype=np.int32)
    nxt = train[user][-1] #user_train
    idx = maxlen - 1
    
    ts = set(train[user]) #user_train
    for i in reversed(train[user][:-1]): #user_train
        seq[idx] = i
        pos[idx] = nxt
        if nxt != 0: neg[idx] = random_neq(1, n_items + 1, ts)
        nxt = i
        idx -= 1
        if idx == -1: break
    
    return (user, seq, pos, neg)


def intermed(train, idxlist, n_users, n_items, maxlen):
    user_l, seq_l, pos_l, neg_l = [], [], [], []
    for user in idxlist:
        new_sample = sample(train, user, n_users, n_items, maxlen)
        user_l.append(new_sample[0])
        seq_l.append(new_sample[1])
        pos_l.append(new_sample[2])
        neg_l.append(new_sample[3])
        
    return user_l, seq_l, pos_l, neg_l

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, config):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        #self.dev = config["device"]

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, config["hidden_units"], padding_idx=0)
        self.pos_emb = torch.nn.Embedding(config["maxlen"], config["hidden_units"]) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p= config["dropout_rate"])

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(config["hidden_units"], eps=1e-8)

        for _ in range(config["num_blocks"]):
            new_attn_layernorm = torch.nn.LayerNorm(config["hidden_units"], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(config["hidden_units"],
                                                            config["num_heads"],
                                                            config["dropout_rate"])
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(config["hidden_units"], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config["hidden_units"], config["dropout_rate"])
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs)) #.to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions)) #.to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0) #.to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool)) #, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs)) #.to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs)) #.to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices)) #.to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
    



def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'



def train_sasrec(config, checkpoint_dir = None, data_dir = None):
    cc = 0.0
    for u in train: #user_train
        cc += len(train[u]) #user_train
    print('average sequence length: %.2f' % (cc / len(train))) #user_train
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            
    model = SASRec(n_users, n_items, config).to(device) # no ReLU activation in original SASRec implementation?
    
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers


    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    #adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    adam_optimizer = torch.optim.Adam(model.parameters(), lr= config["lr"], betas=(0.9, 0.98))
                                      
        
    model.train() # enable model training
    epoch_start_idx = 1
    T = 0.0
    t0 = time.time()
    idxlist = list(range(n_users))
    np.random.shuffle(idxlist)
    users = list(range(n_users))
    
    for epoch in range(epoch_start_idx, config["num_epochs"] + 1):
        for batch_idx, start_idx in enumerate(range(0, n_users, config["batch_size"])): #batchsize
            end_idx = min(start_idx + config["batch_size"], n_users) #batch_size
            #u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = intermed(train, idxlist[start_idx: end_idx], n_users, n_items, config["maxlen"])
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device= device), torch.zeros(neg_logits.shape, device= device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_emb.parameters(): loss += config["l2_emb"] * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, start_idx, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch %  1 == 0: #used to be 20
            model.eval()
            prec100_l, rec100_l, f100_l, hit100_l = [], [], [], []
            ndcg100_l, mrr100_l, arhr100_l, map100_l = [], [], [], []
            div100_l, nov100_l, ser100_l, cc100_l, dc100_l = [], [], [], [], []
            
            for u in users:
                print(u)
                seq = np.zeros([config["maxlen"]], dtype=np.int32)
                idx = config["maxlen"] - 1
                for i in reversed(train_seq[u]): #used to be originally train, just for comparison with bert4rec, see data preprocessing
                    seq[idx] = i
                    idx -= 1
                    if idx == -1: break
        
                recon_batch = model.predict(*[np.array(l) for l in [[u], [seq], cand_val_seq[user]]])
                labels_sas = torch.tensor([1] * len(val[u]) + [0] * 100).unsqueeze(0) #100 is no. of negatives
                prec100_l.append(precision_sequence(recon_batch, labels_sas, 100))
                rec100_l.append(recall_sequence(recon_batch, labels_sas, 100))
                f100_l.append(f1_sequence(scores, labels, 100))
                hit100_l.append(hits_sequence(recon_batch, labels_sas, 100))
                ndcg100_l.append(ndcg_sequence(recon_batch, labels_sas, 100))
                mrr100_l.append(mrr_sequence(recon_batch, labels_sas, 100))
                arhr100_l.append(arhr_sequence(recon_batch, labels_sas, 100))
                map100_l.append(map_sequence(recon_batch, labels_sas, 100))
                mar100_l.append(map_sequence(recon_batch, labels_sas, 100))
                beyond = Beyond_accuracy_prep_sas(recon_batch, train[user] + val[user], 20)
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
            mar_score = np.mean(mar100_l)
            div_score = np.mean(div100_l)
            nov_score = np.mean(nov100_l)
            ser_score = np.mean(ser100_l)
            cc_score = np.mean(cc100_l)
            dc_score = np.mean(dc100_l)
            
        
            tune.report(prec100 = prec_score, rec100 = rec_score, f100 = f1_score, hit100 = hit_score, 
                        ndcg100 = ndcg_score, mrr100 = mrr_score, arhr100 = arhr_score, map100 = map_score, mar100 = mar_score,
                        div100 = div_score, nov100 = nov_score, ser100 = ser_score, cc100 = cc_score, dc100 = dc_score)
"""         
config = {
    "batch_size": tune.choice([1, 2, 4, 8, 16]),
    "lr": tune.choice([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    "maxlen": tune.choice(list(range(100))),#should be the max len seq there is
    "hidden_units": 50,
    "num_blocks": 2, 
    "num_epochs": 5,
    "num_heads": 1,
    "dropout_rate": 0.5,
    "l2_emb": 0.5,
} 
"""
config = {
    "batch_size": tune.choice([1, 2, 4, 8, 16]),
    "lr": tune.choice([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    "maxlen": tune.choice(list(range(100))),#should be the max len seq there is
    "hidden_units": tune.choice([50]),
    "num_blocks": tune.choice([2]), 
    "num_epochs": tune.choice([10]),
    "num_heads": tune.choice([1]),
    "dropout_rate": tune.choice([0.5, 0.75]),
    "l2_emb": tune.choice([0, 0.5, 0.75]),
} 
"""
config = {
    "batch_size": 1,
    "lr": 1e-3,
    "maxlen": 20, #should be the max len seq there is
    "hidden_units": 50,
    "num_blocks": 2, 
    "num_epochs": 10,
    "num_heads": 1,
    "dropout_rate": 0.5,
    "l2_emb": 0,}
"""
     
scheduler = ASHAScheduler(
        metric= "ndcg100",
        mode="max",) 
 
reporter = CLIReporter(metric_columns=["prec100", "rec100", "f100", "hit100",
                                       "ndcg100", "mrr100", "arhr100", "map100", "mar100",
                                       "div100", "nov100", "ser100", "cc100", "dc100"])
                                       

bayesopt = SkOptSearch(metric = "ndcg100", mode = "max")                                  

result_sasrec = tune.run(
    partial(train_sasrec),
    #resources_per_trial={"cpu": 8}, #, "gpu": gpus_per_trial},
    config=config,
    num_samples= 10,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end= False,
    fail_fast = "raise",
    search_alg = bayesopt
    )


best_trial_sasrec = result_sasrec.get_best_trial("ndcg100", "max", "last")
print("Best trial config: {}".format(best_trial_sasrec.config))

print("Best trial final validation precision: {}".format(best_trial_sasrec.last_result["prec100"]))
print("Best trial final validation recall: {}".format(best_trial_sasrec.last_result["rec100"]))
print("Best trial final validation f1: {}".format(best_trial_sasrec.last_result["f100"]))
print("Best trial final validation hits: {}".format(best_trial_sasrec.last_result["hit100"]))

print("Best trial final validation ndcg: {}".format(best_trial_sasrec.last_result["ndcg100"]))
print("Best trial final validation mrr: {}".format(best_trial_sasrec.last_result["mrr100"]))
print("Best trial final validation arhr: {}".format(best_trial_sasrec.last_result["arhr100"]))
print("Best trial final validation map: {}".format(best_trial_sasrec.last_result["map100"]))
print("Best trial final validation mar: {}".format(best_trial_sasrec.last_result["mar100"]))

print("Best trial final validation novelty: {}".format(best_trial_sasrec.last_result["nov100"]))
print("Best trial final validation diversity: {}".format(best_trial_sasrec.last_result["div100"]))
print("Best trial final validation serendipity: {}".format(best_trial_sasrec.last_result["ser100"]))
print("Best trial final validation catalog coverage: {}".format(best_trial_sasrec.last_result["cc100"]))
print("Best trial final validation distributional coverage: {}".format(best_trial_sasrec.last_result["dc100"]))

cc = 0.0
for u in train: #user_train
    cc += len(train[u]) #user_train
print('average sequence length: %.2f' % (cc / len(train))) #user_train
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model = SASRec(n_users, n_items, config).to(device) # no ReLU activation in original SASRec implementation?

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass # just ignore those failed init layers


# this fails embedding init 'Embedding' object has no attribute 'dim'
# model.apply(torch.nn.init.xavier_uniform_)

# ce_criterion = torch.nn.CrossEntropyLoss()
# https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
#adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
adam_optimizer = torch.optim.Adam(model.parameters(), lr= best_trial_sasrec.config["lr"], betas=(0.9, 0.98))
                                  
    
model.train() # enable model training
epoch_start_idx = 1
T = 0.0
t0 = time.time()
np.random.seed(config["SEED"])
idxlist = list(range(n_users))
np.random.shuffle(idxlist)
users = list(range(n_users))

for epoch in range(epoch_start_idx, best_trial_sasrec.last_result["training_iteration"] + 1):
    for batch_idx, start_idx in enumerate(range(0, n_users, best_trial_sasrec.config["batch_size"])): #batchsize
        end_idx = min(start_idx + config["batch_size"], n_users) #batch_size
        #u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
        u, seq, pos, neg = intermed(train, idxlist[start_idx: end_idx], n_users, n_items, best_trial_sasrec.config["maxlen"])
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        pos_logits, neg_logits = model(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device= device), torch.zeros(neg_logits.shape, device= device)
        # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
        adam_optimizer.zero_grad()
        indices = np.where(pos != 0)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        for param in model.item_emb.parameters(): loss += best_trial_sasrec.config["l2_emb"] * torch.norm(param)
        loss.backward()
        adam_optimizer.step()
        print("loss in epoch {} iteration {}: {}".format(epoch, start_idx, loss.item())) # expected 0.4~0.6 after init few epochs

    if epoch %  1 == 0: #used to be 20
        model.eval()
        metrics_n = ["PREC@%d", "REC@%d", "F1@%d", "HIT@%d", "NDCG@%d", "MRR@%d", "ARHR@%d", "MAP@%d", "MAR@%d",
             "DIV@%d", "NOV@%d", "SER@%d", "CC@%d", "DC@%d"]
        ks = [1, 5, 10, 20, 50, 100]
        metrics = {metric_n % k: [] for metric_n in metrics_n for k in ks}
        users = list(range(train_data.shape[0]))

        for u in users:
            seq = np.zeros([best_trial_sasrec.config["maxlen"]], dtype=np.int32)
            idx = best_trial_sasrec.config["maxlen"] - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
    
            recon_batch = model.predict(*[np.array(l) for l in [[u], [seq], cand_val_seq[user]]])
            labels = torch.tensor([1] * len(val[u]) + [0] * 100).unsqueeze(0) #100 is no. of negatives
            
            scores = recon_batch.cpu()
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
                                       
                beyond = Beyond_accuracy_sas(recon_batch.detach().numpy(), train_data[e_idxlist[start_idx: end_idx]] , k) #candidates_test[user].numpy()
                metrics['DIV@%d' % k].append(diversity(beyond[0], beyond[1]))
                metrics['NOV@%d' % k].append(novelty(beyond[0], beyond[1]))
                metrics['SER@%d' % k].append(serendipity(beyond[0], beyond[1]))
                metrics['CC@%d' % k].append(catalog_coverage(beyond[0], beyond[1]))
                metrics['DC@%d' % k].append(distributional_coverage(beyond[0], beyond[1]))
                    

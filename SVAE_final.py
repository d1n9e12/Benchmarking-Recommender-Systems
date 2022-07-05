# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 21:54:01 2022

@author: Daniel
"""
#Taken with courtesy from https://github.com/noveens/svae_cf/blob/master/main_svae.ipynb
#The changes will be reported later on.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import time
import json
import pickle
import random
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import pandas as pd
from scipy import sparse
import numpy as np
  
import bottleneck as bn
import skorch
import sklearn
from torch import optim


import ray    
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune import CLIReporter
from functools import partial




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(
            config['rnn_size'], config['hidden_size']
        )
        nn.init.xavier_normal_(self.linear1.weight) #the deprecation warning
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(config['latent_size'], config['hidden_size'])
        self.linear2 = nn.Linear(config['hidden_size'], config['total_items'])
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        # Since we don't need padding, our vocabulary size = "hyper_params['total_items']" and not "hyper_params['total_items'] + 1"
        self.item_embed = nn.Embedding(config['total_items'], config['item_embed_size'])
        
        self.gru = nn.GRU(
            config['item_embed_size'], config['rnn_size'], 
            batch_first = True, num_layers = 1
        )
        
        self.linear1 = nn.Linear(config['hidden_size'], 2 * config['latent_size'])
        nn.init.xavier_normal_(self.linear1.weight)
        
        self.tanh = nn.Tanh()
        
    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        temp_out = self.linear1(h_enc)
        
        mu = temp_out[:, :self.config['latent_size']]
        log_sigma = temp_out[:, self.config['latent_size']:]
        
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        #if is_cuda_available: std_z = std_z.cuda()

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x):
        in_shape = x.shape                                      # [bsz x seq_len] = [1 x seq_len]
        x = x.view(-1)                                          # [seq_len]
        
        x = self.item_embed(x)                                  # [seq_len x embed_size]
        x = x.view(in_shape[0], in_shape[1], -1)                # [1 x seq_len x embed_size]
        
        rnn_out, _ = self.gru(x)                                # [1 x seq_len x rnn_size]
        rnn_out = rnn_out.reshape(in_shape[0] * in_shape[1], -1)   # [seq_len x rnn_size]
        
        enc_out = self.encoder(rnn_out)                         # [seq_len x hidden_size]
        sampled_z = self.sample_latent(enc_out)                 # [seq_len x latent_size]
        
        dec_out = self.decoder(sampled_z)                       # [seq_len x total_items]
        dec_out = dec_out.view(in_shape[0], in_shape[1], -1)    # [1 x seq_len x total_items]
                              
        return dec_out, self.z_mean, self.z_log_sigma

#ATTENTION, view replaced by reshape as was suggested once 
class VAELoss(torch.nn.Module):
    def __init__(self, config):
        super(VAELoss,self).__init__()
        self.config = config

    def forward(self, decoder_output, mu_q, logvar_q, y_true_s, anneal):
        # Calculate KL Divergence loss
        kld = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q**2 - 1), -1))
    
        # Calculate Likelihood
        dec_shape = decoder_output.shape # [batch_size x seq_len x total_items] = [1 x seq_len x total_items]

        decoder_output = F.log_softmax(decoder_output, -1)
        num_ones = float(torch.sum(y_true_s[0, 0]))
        
        likelihood = torch.sum(
            -1.0 * y_true_s.view(dec_shape[0] * dec_shape[1], -1) * \
            decoder_output.view(dec_shape[0] * dec_shape[1], -1)
        ) / (float(self.config['batch_size']) * num_ones)
        
        final = (anneal * kld) + (likelihood)
        
        return final
   



def train_svae(config, checkpoint_dir = None, data_dir = None):
    model = Model(config)
    device = "cpu"    
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
        
    criterion = VAELoss(config)

    #optimizer = optim.Adam(model.parameters(),  weight_decay = config["wd"]) # config["lr"]
    #optimizer = optim.Adagrad(model.parameters(), weight_decay=config['wd'], lr = ['lr'])
    optimizer = torch.optim.Adadelta(model.parameters(), lr = config["lr"],  weight_decay= config["wd"])
     #   torch.optim.Adam(model.parameters(), weight_decay=hyper_params['weight_decay'])
      #  torch.optim.RMSprop(model.parameters(), weight_decay=hyper_params['weight_decay'])
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)    
    

    for epoch in range(config["epochs"]): 
        model.train()
        total_loss = 0.
        total_anneal_steps = config["total_anneal_steps"]
        anneal = 0.0
        update_count = 0.0
        anneal_cap = config["anneal_cap"]

        start_time = time.time()
        train_epoch_steps = 0
        N = train_data.shape[0]
        idxlist = list(range(N))
        np.random.shuffle(idxlist)    
        for batch_idx, start_idx in enumerate(range(0, N, config["batch_size"])):
            end_idx = min(start_idx + config["batch_size"], N) 
            batch = train_data[idxlist[start_idx:end_idx]]

            batch = torch.LongTensor(batch.toarray())
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch) 
            
            loss = criterion(recon_batch, mu, logvar, batch, anneal)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            print(loss)
            train_epoch_steps += 1
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 
                                1. * update_count / total_anneal_steps) 
            else:
                anneal = anneal_cap
            update_count += 1
    
            """        
            if batch_idx % 100 == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:4d} / {:4d} batches | ms/batch {:4.2f} | '
                        'loss {:4.2f}'.format(
                            epoch, batch_idx, train_epoch_steps,
                            elapsed * 1000 / 100,
                            train_loss / 100)) #see arguments
                
                # Log loss to tensorboard
                #n_iter = (epoch - 1) * len(range(0, N, 100)) + batch_idx #batchsize
                #writer.add_scalars('data/loss', {'train': train_loss / 100}, n_iter) #loginterval
                start_time = time.time()
                train_loss = 0.0
            """
        #Validation loss
        model.eval()
        val_loss = 0.
        val_epoch_steps = 0
        
        prec100_l, rec100_l, f100_l, hit100_l = [], [], [], []
        ndcg100_l, mrr100_l, arhr100_l, map100_l, mar100_l = [], [], [], [], []
        div100_l, nov100_l, ser100_l, cc100_l, dc100_l = [], [], [], [], []
        
        #the following two steps are only for the leave-out-last-one evaluation
        #This was not in the original implementation
        train_seq = variable_sequence_length(train, config["max_len"]) 
        vad_data_tr = build_sparse_simple(train_seq)
        e_N = vad_data_tr.shape[0]
        e_idxlist = list(range(e_N))
        #np.random.shuffle(e_idxlist)

        with torch.no_grad():
            for start_idx in range(0, e_N, config["batch_size"]): #batchsize
                end_idx = min(start_idx + config["batch_size"], e_N)
                tr_batch = vad_data_tr[e_idxlist[start_idx:end_idx]]
                
                batch_tensor = torch.LongTensor(tr_batch.toarray())
                #heldout_batch = vad_data_te[e_idxlist[start_idx:end_idx]]   
                
                recon_batch, mu, logvar = model(batch_tensor)
    
                loss = criterion(recon_batch, mu, logvar, batch_tensor, 0.2) #anneal_cap fixed apparently
                val_loss += loss.item()
                val_epoch_steps += 1    
                # Exclude examples from training set
                #recon_batch = recon_batch.cpu().detach().numpy()
                recon_batch = recon_batch[:, -1, :] # in the same way as in  the original implementation where the authors remove that second dimension in order to compute the last predictions
                recon_batch[tr_batch.nonzero()] = -np.inf 
                prec100_l.append(precision_sequence(recon_batch, labels, 100))
                rec100_l.append(recall_sequence(recon_batch, labels, 100))
                f100_l.append(f1_sequence(recon_batch, labels, 100))
                hit100_l.append(hits_sequence(recon_batch, labels, 100))
                ndcg100_l.append(ndcg_sequence(recon_batch, labels, 100))
                mrr100_l.append(mrr_sequence(recon_batch, labels, 100))
                arhr100_l.append(arhr_sequence(recon_batch, labels, 100))
                map100_l.append(map_sequence(recon_batch, labels, 100))
                mar100_l.append(mar_sequence(recon_batch, labels, 100))
                beyond = Beyond_accuracy_binary(recon_batch.cpu().detach().numpy(), train_data[e_idxlist[start_idx:end_idx]], 100)
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
         
                    
                   

config = {
    "lr": tune.choice([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    "wd": tune.choice([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    "epochs": tune.choice([2]),
    #"batch_size": tune.choice([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]), 
    "batch_size": tune.choice([1]), #due to memory issues
    "item_embed_size": tune.choice([256]),
    "rnn_size": tune.choice([200]),
    "hidden_size": tune.choice([150]),
    "latent_size": tune.choice([64]),
    "number_users_to_keep": tune.choice([1000000000]),
    #'batch_log_interval': 10000,
    "train_cp_users": tune.choice([200]),
    "exploding_clip": tune.choice([0.25]),
    "total_anneal_steps": tune.choice(list(range(10000, 300000, 10000))),
    "anneal_cap": tune.choice(list(np.arange(0, 1, 0.1))),
    "max_len": tune.choice(list(range(1, 200)))
    } 
config["total_items"] = tune.choice([n_items])

"""
#in order to use Bayesian optimization, all parameters have to be included in the form of tune.choice([])
config = {
    "lr": tune.choice([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    "wd": tune.choice([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    "epochs": tune.choice([2]),
    "batch_size": 100, 
    "item_embed_size": 256,
    "rnn_size": 200,
    "hidden_size": 150,
    "latent_size": 64,
    "number_users_to_keep": 100000000,
    #'batch_log_interval': 10000,
    "train_cp_users": 200,
    "exploding_clip": 0.25,
    "total_anneal_steps": 200000,
    "anneal_cap": 0.2,
    "max_len": 50,
    } 
config["total_items"] = n_items
"""
#The usage of the adam optimizer yields a considerable number of Inf or NaN values, try changing 
#it to one of the other optimizers.



scheduler = ASHAScheduler(
        metric= "ndcg100",
        mode="max",) 
  
 
reporter = CLIReporter(metric_columns=["prec100", "rec100", "f100", "hit100",
                                       "ndcg100", "mrr100", "arhr100", "map100", "mar100",
                                       "div100", "nov100", "ser100", "cc100", "dc100",]


bayesopt = SkOptSearch(metric = "ndcg100", mode = "max")                                  

result_svae = tune.run(
    partial(train_svae),
    #resources_per_trial={"cpu": 8}, #, "gpu": gpus_per_trial},
    config=config,
    num_samples= 10,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end= False,
    fail_fast = "raise",
    search_alg = bayesopt
    )


best_trial_svae = result_svae.get_best_trial("ndcg100", "max", "last")
print("Best trial config: {}".format(best_trial_svae.config))

print("Best trial final validation precision: {}".format(best_trial_svae.last_result["prec100"]))
print("Best trial final validation recall: {}".format(best_trial_svae.last_result["rec100"]))
print("Best trial final validation f1: {}".format(best_trial_svae.last_result["f100"]))
print("Best trial final validation hits: {}".format(best_trial_svae.last_result["hit100"]))

print("Best trial final validation ndcg: {}".format(best_trial_svae.last_result["ndcg100"]))
print("Best trial final validation mrr: {}".format(best_trial_svae.last_result["mrr100"]))
print("Best trial final validation arhr: {}".format(best_trial_svae.last_result["arhr100"]))
print("Best trial final validation MAP: {}".format(best_trial_svae.last_result["map100"]))


print("Best trial final validation novelty: {}".format(best_trial_svae.last_result["nov100"]))
print("Best trial final validation diversity: {}".format(best_trial_svae.last_result["div100"]))
print("Best trial final validation serendipity: {}".format(best_trial_svae.last_result["ser100"]))
print("Best trial final validation catalog coverage: {}".format(best_trial_svae.last_result["cc100"]))
print("Best trial final validation distributional coverage: {}".format(best_trial_svae.last_result["dc100"]))


###################################################################################
#MODEL WITH OPTIMAL PARAMETERS
###################################################################################

model = Model(config)
device = "cpu"    
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)
    
criterion = VAELoss(config)

#optimizer = optim.Adam(model.parameters(),  weight_decay = config["wd"]) # config["lr"]
#optimizer = optim.Adagrad(model.parameters(), weight_decay=config['wd'], lr = ['lr'])
optimizer = torch.optim.Adadelta(model.parameters(), lr = config["lr"],  weight_decay= config["wd"])
 #   torch.optim.Adam(model.parameters(), weight_decay=hyper_params['weight_decay'])
  #  torch.optim.RMSprop(model.parameters(), weight_decay=hyper_params['weight_decay'])
if checkpoint_dir:
    model_state, optimizer_state = torch.load(
        os.path.join(checkpoint_dir, "checkpoint"))
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)    


for epoch in range(best_trial_svae.last_result["training_iteration"]): 
    model.train()
    total_loss = 0.
    total_anneal_steps = best_trial_svae.config["total_anneal_steps"]
    anneal = 0.0
    update_count = 0.0
    anneal_cap = best_trial_svae.config["anneal_cap"]

    start_time = time.time()
    train_epoch_steps = 0
    N = train_data.shape[0]
    idxlist = list(range(N))
    np.random.shuffle(idxlist)    
    for batch_idx, start_idx in enumerate(range(0, N, best_trial_svae.config["batch_size"])):
        end_idx = min(start_idx + best_trial_svae.config["batch_size"], N) 
        batch = train_data[idxlist[start_idx:end_idx]]

        batch = torch.LongTensor(batch.toarray())
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch) 
        
        loss = criterion(recon_batch, mu, logvar, batch, anneal)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        print(loss)
        train_epoch_steps += 1
        if total_anneal_steps > 0:
            anneal = min(anneal_cap, 
                            1. * update_count / total_anneal_steps) 
        else:
            anneal = anneal_cap
        update_count += 1

        """        
        if batch_idx % 100 == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d} / {:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, train_epoch_steps,
                        elapsed * 1000 / 100,
                        train_loss / 100)) #see arguments
            
            # Log loss to tensorboard
            #n_iter = (epoch - 1) * len(range(0, N, 100)) + batch_idx #batchsize
            #writer.add_scalars('data/loss', {'train': train_loss / 100}, n_iter) #loginterval
            start_time = time.time()
            train_loss = 0.0
        """
    #Validation loss
    model.eval()
    val_loss = 0.
    val_epoch_steps = 0

    #the following two steps are only for the leave-out-last-one evaluation
    #This was not in the original implementation
    train_seq = variable_sequence_length(train, best_trial_svae.config["max_len"]) 
    vad_data_tr = build_sparse_simple(train_seq)
    e_N = vad_data_tr.shape[0]
    e_idxlist = list(range(e_N))
    metrics_n = ["PREC@%d", "REC@%d", "F1@%d", "HIT@%d", "NDCG@%d", "MRR@%d", "ARHR@%d", "MAP@%d", "MAR@%d",
                 "DIV@%d", "NOV@%d", "SER@%d", "CC@%d", "DC@%d"]
    ks = [1, 5, 10, 20, 50, 100]
    metrics = {metric_n % k: [] for metric_n in metrics_n for k in ks}

    with torch.no_grad():
        for start_idx in range(0, e_N, best_trial_svae.config["batch_size"]): #batchsize
            end_idx = min(start_idx + best_trial_svae.config["batch_size"], e_N)
            tr_batch = vad_data_tr[e_idxlist[start_idx:end_idx]]
            
            batch_tensor = torch.LongTensor(tr_batch.toarray())
            #heldout_batch = vad_data_te[e_idxlist[start_idx:end_idx]]   
            labels = labels_val[e_idxlist[start_idx: end_idx]]
            
            recon_batch, mu, logvar = model(batch_tensor)

            loss = criterion(recon_batch, mu, logvar, batch_tensor, 0.2) #anneal_cap fixed apparently
            val_loss += loss.item()
            val_epoch_steps += 1    
            # Exclude examples from training set
            #recon_batch = recon_batch.cpu().detach().numpy()
            recon_batch = recon_batch[:, -1, :] # in the same way as in  the original implementation where the authors remove that second dimension in order to compute the last predictions
            recon_batch[tr_batch.nonzero()] = -np.inf 
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
                metrics['HIT@%d' % k].append(np.mean([np.any(hit) for hit in hits.numpy()]))
                
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
                mrr = np.mean(rank_l)
                metrics['MRR@%d' % k].append(mrr)
        
                position_arhr = torch.arange(1, k + 1)
                weights_arhr = 1 / position_arhr.float()
                arhr = (hits * weights_arhr).sum(1)
                arhr = arhr.mean().item()
                metrics['ARHR@%d' % k].append(arhr)
        
                p_at_k = hits.numpy() * np.cumsum(hits.numpy(), axis = 1, dtype=np.float64) / (1 + np.arange(hits.numpy().shape[1])) #look for explanation beneath
                a_p = torch.sum(p_at_k, dim = 1) / torch.min(torch.Tensor([k]), labels.sum(1).float())
                metrics['MAP@%d' % k].append(a_p.mean().item())
                
                a_r = torch.sum(p_at_k, dim = 1) / labels.sum(1).float()  #batch_size results in 1, doesn't seem useful, take shape[1] because it's about the length of the list, but here it's a 2dim array
                metrics['MAR@%d' % k].append(a_r.mean().item())
                                    
                beyond = Beyond_accuracy_binary(recon_batch.detach().numpy(), train_data[e_idxlist[start_idx: end_idx]] , k) #candidates_test[user].numpy()
                metrics['DIV@%d' % k].append(diversity(beyond[0], beyond[1]))
                metrics['NOV@%d' % k].append(novelty(beyond[0], beyond[1]))
                metrics['SER@%d' % k].append(serendipity(beyond[0], beyond[1]))
                metrics['CC@%d' % k].append(catalog_coverage(beyond[0], beyond[1]))
                metrics['DC@%d' % k].append(distributional_coverage(beyond[0], beyond[1]))
                    
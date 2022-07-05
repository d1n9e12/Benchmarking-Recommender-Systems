# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:53:25 2022

@author: Daniel
"""
#taken with courtesy from https://github.com/younggyoseo/vae-cf-pytorch and slightly adapted
#changes will be stated later on
import pandas as pd
from scipy import sparse
import numpy as np
import os
  
#import bottleneck as bn
from torch import optim
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune import CLIReporter
from functools import partial



"""
def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]
    
    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]
    
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId')
    return tp, usercount, itemcount


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])
        
        else:
            tr_list.append(group)
        
    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    uid = tp['userId'].apply(lambda x: profile2id[x])
    sid = tp['movieId'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


print("Load and Preprocess Movielens-20m dataset")
# Load Data
DATA_DIR = 'C:/Users/Daniel/Downloads/ml-1m/'
#raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header=0)
raw_data = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'), header= None,
                    names =  ["userId", "movieId", "rating", "timestamp"], sep = "::")
raw_data = raw_data[raw_data['rating'] > 3.5]

# Filter Data
raw_data, user_activity, item_popularity = filter_triplets(raw_data)

# Shuffle User Indices
unique_uid = user_activity.index
np.random.seed(98765)
idx_perm = np.random.permutation(unique_uid.size)
unique_uid = unique_uid[idx_perm]

n_users = unique_uid.size
n_heldout_users = 1000

# Split Train/Validation/Test User Indices
tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
te_users = unique_uid[(n_users - n_heldout_users):]

train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]
unique_sid = pd.unique(train_plays['movieId'])

show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

pro_dir = os.path.join(DATA_DIR, 'pro_sg')

if not os.path.exists(pro_dir):
    os.makedirs(pro_dir)

with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)

vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)]
vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)]

vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]
test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

train_data = numerize(train_plays, profile2id, show2id)
train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

vad_data_te = numerize(vad_plays_te, profile2id, show2id)
vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

test_data_tr = numerize(test_plays_tr, profile2id, show2id)
test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

test_data_te = numerize(test_plays_te, profile2id, show2id)
test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

print("Done!")


#This operation amounts to the same thing apparently
n_users = train_data['uid'].max() + 1 #WHATTTTTTTTTTTTTTTTTTTTTTTTTTT???????????????????
n_users_tr = pd.unique(train_data["uid"]).shape[0]

n_items = pd.unique(raw_data["movieId"])
n_items = len(show2id)
n_items = len(unique_sid)  


train_data = sparse.csr_matrix((np.ones_like(train_data["uid"]),
                         (train_data["uid"], train_data["sid"])), dtype='float64', shape=(n_users_tr, n_items))

#These 2 steps don't work
"""
start_idx = min(vad_data_tr_['uid'].min(), vad_data_te_['uid'].min())
end_idx = max(vad_data_tr_['uid'].max(), vad_data_te_['uid'].max())

rows_tr, cols_tr = vad_data_tr_['uid'] - start_idx, vad_data_tr_['sid']
rows_te, cols_te = vad_data_te_['uid'] - start_idx, vad_data_te_['sid']

vad_data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                            (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
"""
#so this is used instead
n_users_vd_tr =vad_data_tr["uid"].max() + 1
#n_users_vd_tr =  pd.unique(vad_data_tr["uid"]).shape[0]
vad_data_tr = sparse.csr_matrix((np.ones_like(vad_data_tr["uid"]),
                            (vad_data_tr["uid"], vad_data_tr["sid"])), dtype='float64', shape=(n_users_vd_tr, n_items))

n_users_vd_te =vad_data_te["uid"].max() + 1


vad_data_te = sparse.csr_matrix((np.ones_like(vad_data_te["uid"]),
                            (vad_data_te["uid"], vad_data_te["sid"])), dtype='float64', shape=(n_users_vd_te, n_items))


n_users_te_tr = test_data_tr["uid"].max() + 1
#n_users_test_te =  pd.unique(test_data_tr_["uid"]).shape[0]

test_data_te = sparse.csr_matrix((np.ones_like(test_data_te["uid"]),
                            (test_data_te["uid"], test_data_te["sid"])), dtype='float64', shape=(n_users_te_tr, n_items))

test_data_tr = sparse.csr_matrix((np.ones_like(test_data_tr["uid"]),
                            (test_data_tr["uid"], test_data_tr["sid"])), dtype='float64', shape=(n_users_te_tr, n_items))

vad_data_tr = vad_data_tr[4034:]
vad_data_te = vad_data_te[4034:]
test_data_tr = test_data_tr[4034:]
test_data_te = test_data_te[4034:]

#train_data = train_data[:1000]
vad_plays.reset_index(drop = True)
vad_plays.columns = ["uid", "sid", "rating", "timestamp"]
vad_group = vad_plays.groupby('uid')
vad_user2items = vad_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))

vad_labels = []
for index, (user, items) in enumerate(vad_user2items.items()):
    labels = np.zeros(n_items, dtype=np.float) #mask_token is same
    labels[index - 1] += 1
    vad_labels.append(labels)

test_plays.reset_index(drop = True)
test_plays.columns = ["uid", "sid", "rating", "timestamp"]
test_group = test_plays.groupby('uid')
test_user2items = test_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))

test_labels = []
for user, items in test_user2items.items():
    labels = np.zeros(n_items, dtype=np.float) #mask_token is same
    for item in items:
        labels[item - 1] += 1
        test_labels.append(labels)
    
    
    
    
del raw_data
del train_plays
del vad_plays
del vad_plays_tr
del vad_plays_te
del test_plays
del test_plays_tr
del test_plays_te
"""
################################################################################
################################################################################
################################################################################
################################################################################

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h) #F.tanh
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h) #F.tanch #got warning that this was depecracted
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD

def train_mvae(config, checkpoint_dir = None):
    p_dims = [200, 600, config["n_items"]]
    model = MultiVAE(p_dims)
    
    device = "cpu"    
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
        
    criterion = loss_function   
    optimizer = optim.Adam(model.parameters(), lr= config["lr"], weight_decay= config["wd"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)    
    
    for epoch in range(config["epochs"]): 
        update_count = 0
        total_anneal_steps = config["total_anneal_steps"]
        anneal_cap = config["anneal_cap"]
        model.train()
        train_loss = 0.
        start_time = time.time()
        
        train_epoch_steps = 0
        N = train_data.shape[0]
        idxlist = list(range(N))
        np.random.shuffle(idxlist)    
        for batch_idx, start_idx in enumerate(range(0, N, config["batch_size"])): 
            end_idx = min(start_idx + config["batch_size"], N) 
            batch = train_data[idxlist[start_idx:end_idx]]
            
            batch = naive_sparse2tensor(batch).to(device)        
        
            if total_anneal_steps > 0: 
                anneal = min(anneal_cap, 
                                1. * update_count / total_anneal_steps) 
            else:
                anneal = anneal_cap
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch.to(device)) #once already suffices?
            
            loss = criterion(recon_batch, batch, mu, logvar, anneal)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()  #should this be behind train_loss?
        
            update_count += 1  
            train_epoch_steps += 1
        
            """
            if batch_idx % 100 == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:4d} / {:4d} batches | ms/batch {:4.2f} | '
                        'loss {:4.2f}'.format(
                            epoch, batch_idx, train_epoch_steps,
                            elapsed * 1000 / 100,
                            train_loss / 100)) #see arguments

                start_time = time.time()
                train_loss = 0.0
            """
        #Validation 
        model.eval()
        total_loss = 0.
        total_anneal_steps = config["total_anneal_steps"]
        anneal_cap = config["anneal_cap"]
        prec100_l, rec100_l, f100_l, hit100_l = [], [], [], []
        ndcg100_l, mrr100_l, arhr100_l, map100_l, mar100_l = [], [], [], [], []
        div100_l, nov100_l, ser100_l, cc100_l, dc100_l = [], [], [], [], []
       
        val_epoch_steps = 0
        #the following two steps are only for the leave-out-last-one evaluation
        #This was not in the original implementation.
        train_seq = variable_sequence_length(train, config["max_len"]) 
        vad_data_tr = build_sparse_simple(train_seq)
        e_N = vad_data_tr.shape[0]
        e_idxlist = list(range(e_N))

        with torch.no_grad():
            for start_idx in range(0, e_N, config["batch_size"]): 
                end_idx = min(start_idx + config["batch_size"],  e_N)
                tr_batch = vad_data_tr[e_idxlist[start_idx:end_idx]]  #vad_data_tr
                batch_tensor = naive_sparse2tensor(tr_batch)
                #heldout_batch = vad_data_te[e_idxlist[start_idx:end_idx]] 
                labels = labels_val[e_idxlist[start_idx: end_idx]]
                
                if total_anneal_steps > 0: 
                    anneal = min(anneal_cap, 
                                1. * update_count / total_anneal_steps) 
                else:
                    anneal = anneal_cap
                
                recon_batch, mu, logvar = model(batch_tensor)
    
                loss = criterion(recon_batch, batch_tensor, mu, logvar, anneal)
                total_loss += loss.item()
                print(loss)
                val_epoch_steps += 1
    
                # Exclude examples from training set
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


                beyond = Beyond_accuracy_binary(recon_batch.cpu().detach().numpy(),
                                                train_data[e_idxlist[start_idx:end_idx]] , 100)
                #REMEMBER, if holdout, use only the VAD_DATA_TR
                div100_l.append(diversity(beyond[0], beyond[1]))
                nov100_l.append(novelty(beyond[0], beyond[1]))
                ser100_l.append(serendipity(beyond[0], beyond[1]))
                cc100_l.append(catalog_coverage(beyond[0], beyond[1]))
                dc100_l.append(distributional_coverage(beyond[0], beyond[1]))
                
                
        
            prec_score = np.mean(prec100_l)    
            rec_score = np.mean(rec100_l)
            hit_score = np.mean(hit100_l)
            ndcg_score = np.mean(ndcg100_l)
            mrr_score = np.mean(mrr100_l)
            arhr_score = np.mean(arhr100_l)
            map_score = np.mean(map100_l)
            mar_score  np.mean(mar100_l)
            div_score = np.mean(div100_l)
            nov_score = np.mean(nov100_l)
            ser_score = np.mean(ser100_l)
            cc_score = np.mean(cc100_l)
            dc_score = np.mean(dc100_l)
            
        
            tune.report(prec100 = prec_score, rec100 = rec_score, f100 = f1_score, hit100 = hit_score, 
                        ndcg100 = ndcg_score, mrr100 = mrr_score, arhr100 = arhr_score, map100 = map_score, mar100 = mar_score,
                        div100 = div_score, nov100 = nov_score, ser100 = ser_score, cc100 = cc_score, dc100 = dc_score)
         
item_lengths = []
for user, items in train.items():
    item_lengths.append(len(items))  
np.median(item_lengths) #is 56, just took 200 as           

#Dimension of the latent representation K to 200 and hidden layer to 600        
config = {"lr" : tune.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
          "wd" : tune.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
          "batch_size" : tune.choice([20, 50, 100, 200]),
          "epochs": tune.choice([10]),
          "total_anneal_steps": tune.choice(list(range(0, 300000, 10000))),
          "anneal_cap": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
          "max_len": tune.choice(list(range(1,200 + 1)))}
config["n_items"] = tune.choice([n_items])
"""
#just a test config, if you want to merely run through the model by chunks of code
config = {"lr" : (1e-10, 1e-1),
          "wd" : (1e-10, 1e-1),
          "batch_size" :(10, 500),
          "epochs": 10,
          "total_anneal_steps": (1000, 3000000),
          "anneal_cap": (0, 1),
          "max_len": (1, 201)}
config["n_items"] = n_items
"""
scheduler = ASHAScheduler(
        metric= "ndcg100",
        mode="max",) 
 
reporter = CLIReporter(metric_columns=["prec100", "rec100", "f100", "hit100",
                                       "ndcg100", "mrr100", "arhr100", "map100", "mar100"
                                       "div100", "nov100", "ser100", "cc100", "dc100",
                                       "time_total_s", "training_iteration"])

bayesopt = SkOptSearch(metric = "ndcg100", mode = "max")                                  

result_mvae = tune.run(
    partial(train_mvae),
    #resources_per_trial={"cpu": 8}, #, "gpu": gpus_per_trial},
    config=config,
    num_samples= 10,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end= False,
    fail_fast = "raise",
    search_alg = bayesopt
    )

best_trial_mvae = result_mvae.get_best_trial("ndcg100", "max", "last")
print("Best trial config: {}".format(best_trial_mvae.config))

print("Best trial final validation precision: {}".format(best_trial_mvae.last_result["prec100"]))
print("Best trial final validation recall: {}".format(best_trial_mvae.last_result["rec100"]))
print("Best trial final validation hits: {}".format(best_trial_mvae.last_result["hit100"]))

print("Best trial final validation ndcg: {}".format(best_trial_mvae.last_result["ndcg100"]))
print("Best trial final validation mrr: {}".format(best_trial_mvae.last_result["mrr100"]))
print("Best trial final validation arhr: {}".format(best_trial_mvae.last_result["arhr100"]))
print("Best trial final validation map: {}".format(best_trial_mvae.last_result["map100"]))
print("Best trial final validation mar: {}".format(best_trial_mvae.last_result["mar100"]))


print("Best trial final validation novelty: {}".format(best_trial_mvae.last_result["nov100"]))
print("Best trial final validation diversity: {}".format(best_trial_mvae.last_result["div100"]))
print("Best trial final validation serendipity: {}".format(best_trial_mvae.last_result["ser100"]))
print("Best trial final validation catalog coverage: {}".format(best_trial_mvae.last_result["cc100"]))
print("Best trial final validation distributional coverage: {}".format(best_trial_mvae.last_result["dc100"]))

###################################################################################
#MODEL WITH OPTIMAL PARAMETERS
###################################################################################


p_dims = [200, 600, best_trial_mvae.config["n_items"]]
model = MultiVAE(p_dims)

device = "cpu"    
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)
    
criterion = loss_function   
optimizer = optim.Adam(model.parameters(), lr= best_trial_mvae.config["lr"], weight_decay= best_trial_mvae.config["wd"])
checkpoint_dir = None
if checkpoint_dir:
    model_state, optimizer_state = torch.load(
        os.path.join(checkpoint_dir, "checkpoint"))
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)    

for epoch in range(best_trial_mvae.last_result["training_iteration"]): 
    update_count = 0
    total_anneal_steps = best_trial_mvae.config["total_anneal_steps"]
    anneal_cap = best_trial_mvae.config["anneal_cap"]
    model.train()
    train_loss = 0.
    start_time = time.time()
    
    train_epoch_steps = 0
    N = train_data.shape[0]
    idxlist = list(range(N))
    np.random.shuffle(idxlist)    
    for batch_idx, start_idx in enumerate(range(0, N, best_trial_mvae.config["batch_size"])): 
        end_idx = min(start_idx + best_trial_mvae.config["batch_size"], N) 
        batch = train_data[idxlist[start_idx:end_idx]]
        
        batch = naive_sparse2tensor(batch).to(device)        
    
        if total_anneal_steps > 0: 
            anneal = min(anneal_cap, 
                            1. * update_count / total_anneal_steps) 
        else:
            anneal = anneal_cap
        
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch.to(device)) #once already suffices?
        
        loss = criterion(recon_batch, batch, mu, logvar, anneal)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()  #should this be behind train_loss?
    
        update_count += 1  
        train_epoch_steps += 1
    
        """
        if batch_idx % 100 == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:4d} / {:4d} batches | ms/batch {:4.2f} | '
                    'loss {:4.2f}'.format(
                        epoch, batch_idx, train_epoch_steps,
                        elapsed * 1000 / 100,
                        train_loss / 100)) #see arguments

            start_time = time.time()
            train_loss = 0.0
        """
    #Validation 
    model.eval()
    total_loss = 0.
    total_anneal_steps = best_trial_mvae.config["total_anneal_steps"]
    anneal_cap = best_trial_mvae.config["anneal_cap"]

    val_epoch_steps = 0
    #the following two steps are only for the leave-out-last-one evaluation
    #This was not in the original implementation.
    train_seq = variable_sequence_length(train, best_trial_mvae.config["max_len"]) 
    vad_data_tr = build_sparse_simple(train_seq)
    e_N = vad_data_tr.shape[0]
    e_idxlist = list(range(e_N))
    metrics_n = ["PREC@%d", "REC@%d", "F1@%d", "HIT@%d", "NDCG@%d", "MRR@%d", "ARHR@%d", "MAP@%d", "MAR@%d",
                 "DIV@%d", "NOV@%d", "SER@%d", "CC@%d", "DC@%d"]
    ks = [1, 5, 10, 20, 50, 100]
    metrics = {metric_n % k: [] for metric_n in metrics_n for k in ks}
    with torch.no_grad():
        for start_idx in range(0, e_N, best_trial_mvae.config["batch_size"]): 
            end_idx = min(start_idx + best_trial_mvae.config["batch_size"],  e_N)
            tr_batch = vad_data_tr[e_idxlist[start_idx:end_idx]]  
            batch_tensor = naive_sparse2tensor(tr_batch)
            #heldout_batch = vad_data_te[e_idxlist[start_idx:end_idx]] 
            labels = labels_test[e_idxlist[start_idx: end_idx]]
            
            if total_anneal_steps > 0: 
                anneal = min(anneal_cap, 
                            1. * update_count / total_anneal_steps) 
            else:
                anneal = anneal_cap
            
            recon_batch, mu, logvar = model(batch_tensor)

            loss = criterion(recon_batch, batch_tensor, mu, logvar, anneal)
            total_loss += loss.item()
            print(loss)
            val_epoch_steps += 1

            # Exclude examples from training set
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
                    
                beyond = Beyond_accuracy_binary(recon_batch.detach().numpy(), train_data[e_idxlist[start_idx: end_idx]] , k) #candidates_test[user].numpy()
                metrics['DIV@%d' % k].append(diversity(beyond[0], beyond[1]))
                metrics['NOV@%d' % k].append(novelty(beyond[0], beyond[1]))
                metrics['SER@%d' % k].append(serendipity(beyond[0], beyond[1]))
                metrics['CC@%d' % k].append(catalog_coverage(beyond[0], beyond[1]))
                metrics['DC@%d' % k].append(distributional_coverage(beyond[0], beyond[1]))
                    

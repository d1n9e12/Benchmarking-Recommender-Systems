# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:23:24 2021

@author: Daniel
"""

"""
#Here's how to drop the cold users!!!!!!!!!!!!!!
def df_to_sparse(df, shape):
    
    Utility function to transform raw data frame to sparse matrix
    :param df: raw data frame
    :param shape: shape of the sparse matrix
    :return: new sparse data frame
    
    df = train_df
    rows, cols = df.reviewerID, df.asin
    values = df.overall #np.ones

    sp_data = sp.csr_matrix((values, (rows, cols)), dtype='float32', shape=shape) #originally float64
    
    num_nonzeros = np.diff(sp_data.indptr)
    rows_to_drop = num_nonzeros == 0
    if sum(rows_to_drop) > 0:
        print('%d empty users are dropped from matrix.' % sum(rows_to_drop))
        sp_data = sp_data[num_nonzeros != 0]
    
    return sp_data


def sparse_to_dict(sparse_matrix):
        
        #Function to convert sparse data matrix to a dictionary
        #:param sparse_matrix: sparse data matrix
        #:return: dictionary to hold the data
        
        ret_dict = {}
        num_users = sparse_matrix.shape[0]
        for u in range(num_users):
            items_u = sparse_matrix.indices[sparse_matrix.indptr[u]: sparse_matrix.indptr[u + 1]]
            ret_dict[u] = items_u.tolist()
        return ret_dict

   
"""
##################################################################
#PREPROCESSING OF THE DATASET
##################################################################
#imports


import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm
tqdm.pandas()

from abc import *
from pathlib import Path
import dill as pickle
import random
import torch.backends.cudnn as cudnn
from abc import ABCMeta, abstractmethod
from tqdm import trange
import torch
"""
path =  "C:/Users/Daniel/Downloads/amazon/"
df = pd.read_json(path + "CD.json", lines = True)
#duplicate_bool = df.duplicated(subset = ["reviewerID", "reviewText", "overall", "unixReviewTime"])

df.drop_duplicates(subset = ["reviewerID", "reviewText", "overall", "unixReviewTime"],
                    keep = "first", inplace = True)

#df2 = pd.read_csv(path + "ratings.csv", sep = ",")


df = df[["reviewerID", "asin", "overall", "unixReviewTime"]]
df.columns = ["uid", "sid", "rating", "timestamp"]
"""
path =  "C:/Users/Daniel/Downloads/ml-1m/"
df = pd.read_csv(path + "ratings.csv", sep= "::", names = ["uid", "sid", "rating", "timestamp"])


#The following preprocessing code has been taken from https://github.com/khanhnamle1994/MetaRec/blob/master/Autoencoders-Experiments/VAE-PyTorch/DataUtils.py
#and adapted 
num_users = len(pd.unique(df.uid))
num_items = len(pd.unique(df.sid))


 # Convert ratings into implicit feedback
#df['rating'] = 1.0

num_items_by_user = df.groupby('uid', as_index=False).size()
num_users_by_item = df.groupby('sid', as_index=False).size()           

 # Assign new user IDs
print('Assign new user id...')
user_frame = num_items_by_user.to_frame()
user_frame.columns = ['item_cnt']

order_by_popularity=True

if order_by_popularity:
    user_frame = user_frame.sort_values(by='item_cnt', ascending=False)
user_frame['new_id'] = list(range(num_users))

# Add old user IDs into new consecutive user IDs
frame_dict = user_frame.to_dict()
user_id_dict = frame_dict['new_id']
user_frame = user_frame.set_index('new_id')
user_to_num_items = user_frame.to_dict()['item_cnt']

df.uid = [user_id_dict[x] for x in df.uid.tolist()]      

# Assign new item IDs
item_frame = num_users_by_item.to_frame()
item_frame.columns = ['user_cnt']

if order_by_popularity:
    item_frame = item_frame.sort_values(by='user_cnt', ascending=False)
item_frame['new_id'] = range(num_items)

    # Add old item IDs into new consecutive item IDs
frame_dict = item_frame.to_dict()
item_id_dict = frame_dict['new_id']
item_frame = item_frame.set_index('new_id')
item_to_num_users = item_frame.to_dict()['user_cnt']  
df.sid = [item_id_dict[x] for x in df.sid.tolist()]

num_users, num_items = len(user_id_dict), len(item_id_dict)
num_ratings = len(df)
######################################### end of the code
#The following code has been taken from https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch/tree/master/datasets
#and adapted
df = df.sort_values(["uid", "timestamp"])
df = df.reset_index(drop = True)

df = df[df['rating'] >=4] # 
# return df[['uid', 'sid', 'timestamp']]
  #  return df

#def filter_triplets(df):
print('Filtering triplets')
#if 2 > 0:
item_sizes = df.groupby('sid').size() #how many users does an item have?
good_items = item_sizes.index[item_sizes >= 1] 
df = df[df['sid'].isin(good_items)]

#if 2 > 0:
user_sizes = df.groupby('uid').size() #how many items does a user have
good_users = user_sizes.index[user_sizes >= 5]
df = df[df['uid'].isin(good_users)]



#def densify_index(self, df):
print('Densifying index')
umap = {u: i for i, u in enumerate(set(df['uid']))} #for index, user in ... map user to index
smap = {s: i for i, s in enumerate(set(df['sid']))} #for index, item in ... map item to index
df['uid'] = df['uid'].map(umap)
df['sid'] = df['sid'].map(smap)
 #   return df, umap, smap


#def split_df(self, df, user_count):
#   if self.args.split == 'leave_one_out':
print('Splitting')
user_group = df.groupby('uid')
user2items = user_group.progress_apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
train, val, test = {}, {}, {}
for user in range(len(df["uid"].unique())):
    items = user2items[user]
    train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]

import torch.utils.data as data_utils
def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

  
#class BertDataloader(AbstractDataloader):
 #   def __init__(self, args, dataset):
        #super().__init__(args, dataset)
num_items = n_items = len(smap)
n_users = len(umap)
seed = fix_random_seed_as(1)
max_len = 20 #args.bert_max_len
#mask_prob = 0.02 #args.bert_mask_prob
item_count = len(smap)
#BEWARE THOSE ONES AT THE END!!!, ORIginally for bert and sasrec they started sequences from 1 not zero!
CLOZE_MASK_TOKEN = item_count# + 1 
rng = random.Random(seed)

class AbstractNegativeSampler(metaclass=ABCMeta):
    def __init__(self, train, val, test, user_count, item_count, sample_size, seed, save_folder):
        self.train = train
        self.val = val
        self.test = test
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size
        self.seed = seed
        self.save_folder = save_folder

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def generate_negative_samples(self):
        pass

    def get_negative_samples(self):
        savefile_path = self._get_save_path()
        if savefile_path.is_file():
            print('Negatives samples exist. Loading.')
            negative_samples = pickle.load(savefile_path.open('rb'))
            return negative_samples
        print("Negative samples don't exist. Generating.")
        negative_samples = self.generate_negative_samples()
        with savefile_path.open('wb') as f:
            pickle.dump(negative_samples, f)
        return negative_samples

    def _get_save_path(self):
        folder = Path(self.save_folder)
        filename = '{}-sample_size{}-seed{}.pkl'.format(self.code(), self.sample_size, self.seed)
        return folder.joinpath(filename)
    
#Here, because of the preprocessing and the csr-format, the item range is from 0 -> n_items
#However, sequential recommenders demand that this be from 1 -> n_items + 1 because of the zeros added before the first item in the generated sequences which are then fed to the model
#This means that the negatives here will be tailored to the the csr-format recommenders algorithms 
#For the sequential recommenders, to each item generated will be added a one.
class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        negative_samples = {}
        print('Sampling negative items')
        for user in trange(self.user_count):
            if isinstance(self.train[user][1], tuple):
                seen = set(x[0] for x in self.train[user])
                seen.update(x[0] for x in self.val[user])
                seen.update(x[0] for x in self.test[user])
            else:
                seen = set(self.train[user])
                seen.update(self.val[user])
                seen.update(self.test[user])

            samples = []
            for _ in range(self.sample_size):
                item = np.random.choice(self.item_count)  # + 1
                while item in seen or item in samples:
                    item = np.random.choice(self.item_count) #+ 1 
                samples.append(item)

            negative_samples[user] = samples

        return negative_samples    
    
    
train_negative_sampler = RandomNegativeSampler(train, val, test,
                                              len(df["uid"].unique()), item_count,
                                              100, #args.train_negative_sample_size,
                                             1, # args.train_negative_sampling_seed,
                                              path)
test_negative_sampler = RandomNegativeSampler( train, val, test,
                                             len(df["uid"].unique()), item_count, #len(umap)
                                             100 ,#args.test_negative_sample_size,
                                             1, #args.test_negative_sampling_seed,
                                             path)

train_negative_samples = train_negative_sampler.get_negative_samples()
test_negative_samples = test_negative_sampler.get_negative_samples()

#WIth courtesy from the programmers from the https://github.com/sisinflab/elliot/blob/master/elliot/dataset/dataset.py
#and adapted because the original function is called build_sparse, and its input was a dictionary
# takes into account the items as values with their respective ratings
def build_sparse_simple(dict_): #just train_dict, no nested dictionaries

    rows_cols = [(u, i) for u, items in dict_.items() for i in items]
    rows = [u for u, _ in rows_cols]
    cols = [i for _, i in rows_cols]
    data = sps.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                         shape=(len(umap), len(smap))) #umap, #smap
    return data

train_data = build_sparse_simple(train)

#Given the current evaluation method with the torch.gather(1, cut) option
#The following remains until, the Beyond_accuracy_prep is adapted specifically to take into account
#labels whose first dimension is larger than 1
val_data = build_sparse_simple(val)
train_negatives = build_sparse_simple(train_negative_samples)
val_candidate_items = val_data + train_negatives
test_data = build_sparse_simple(test)
test_negatives = build_sparse_simple(test_negative_samples)
test_candidate_items = test_data + test_negatives

###########################################################
#This code is taken from the BERT4REC pytorch implementation from https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch/blob/master/dataloaders/bert.py
#and adapted
#VAL AND TEST SET 
#Next to the csr-format matrices, torch.LongTensort which contain the labels are generated for the specifically for the evaluation method
#used in the accuracy metrics: torch.gather(1, cut)
u2seq = train
users = sorted(u2seq.keys())
u2answer = val
#max_len = max_len
#mask_token = CLOZE_MASK_TOKEN
negative_samples = train_negative_samples


#def __len__(self):
 #   return len(users)

#def __getitem__(self, index):
cand_l_v, labels_l_v = [], []
for index in range(len(users)):
    user = users[index]
    seq = u2seq[user]
    answer = u2answer[user]
    negs = negative_samples[user]

    candidates = answer + negs

    #labels = [1] * len(answer) + [0] * (mask_token - 1) 
    labels = np.zeros(n_items, dtype=np.float) #mask_token is same
    labels[candidates] += 1
    
    
    #seq = seq + [mask_token - 1]
    #seq = seq[-max_len:]
    #padding_len = max_len - len(seq)
    #seq = [0] * padding_len + seq
    
    #seq_l_v.append(seq)
    cand_l_v.append(candidates)
    labels_l_v.append(labels)

candidates_val, labels_val =  torch.LongTensor(cand_l_v), torch.LongTensor(labels_l_v) 

"""
seq_val_dict_ = {user: items for user, items in enumerate(seq_l_v)}    
seq_val_data = build_sparse_simple(seq_val_dict_) #so, the mask_token can be reduced by 1, or train without it
vad_data_tr= seq_val_data
#BUt remember, where in BERT all itmes are moved by that "1", see to this.
"""
#VAL AND TEST SET BERT
u2seq = train
users = sorted(u2seq.keys())
u2answer = test
#max_len = max_len
#mask_token = CLOZE_MASK_TOKEN
negative_samples = test_negative_samples


#def __len__(self):
 #   return len(users)

#def __getitem__(self, index):
cand_l_t, labels_l_t = [], []
for index in range(len(users)):
    user = users[index]
    seq = u2seq[user]
    answer = u2answer[user]
    negs = negative_samples[user]

    candidates = answer + negs

    #labels = [1] * len(answer) + [0] * (mask_token - 1) 
    labels = np.zeros(n_items, dtype=np.float) #mask_token is same
    labels[candidates] += 1
    
    
    #seq = seq + [mask_token - 1]
    #seq = seq[-max_len:]
    #padding_len = max_len - len(seq)
    #seq = [0] * padding_len + seq
    
    #seq_l_t.append(seq)
    cand_l_t.append(candidates)
    labels_l_t.append(labels)

candidates_test, labels_test = torch.LongTensor(cand_l_t), torch.LongTensor(labels_l_t) 
"""
seq_test_dict_ = {user: items for user, items in enumerate(seq_l_t)}    
seq_test_data = build_sparse_simple(seq_test_dict_) #so, the mask_token can be reduced by 1, or train without it
"""
def variable_sequence_length(train, maxlen):
    seq_dict = {}
    for user, items in train.items():
        seq_dict[user] = items[-maxlen:]
    return seq_dict

#FOR sequential recommenders such as Sasrec, not necessary for Bert4rec
#because the model uses sequences determined by the maximum length as a config parameter,
#if the maximum length designated to be larger than the number of items rated, it will append zeros
#before the items. However, the first item is item zero instead of one (has to be for the csr-format), so that's why all items are increased by one
train_seq = {}
for user, items in train.items():
    items_l = []
    for item in items:
        items_l.append(item + 1)
    train_seq[user] = items_l

val_seq = {}
for user, items in val.items():
    items_l = []
    for item in items:
        items_l.append(item + 1)
    val_seq[user] = items_l
    
test_seq = {}
for user, items in test.items():
    items_l = []
    for item in items:
        items_l.append(item + 1)
    test_seq[user] = items_l
   
  
cand_val_seq = []
for items in cand_l_v:
    user_l = []
    for item in items:
        user_l.append(item + 1)
    cand_val_seq.append(user_l)

cand_test_seq = []
for items in cand_l_t:
    user_l = []
    for item in items:
        user_l.append(item + 1)
    cand_test_seq.append(user_l)    
    
##
#USE COMMON IDXLIST TO SHUFFLE THE ROWS

"""
THE HOLD-OUT STRATEGY
np.random.seed(1)
eval_set_size = 1000

# Generate user indices
permuted_index = np.random.permutation(n_users)
train_user_index = permuted_index[                :-2*eval_set_size]
val_user_index   = permuted_index[-2*eval_set_size:  -eval_set_size]
test_user_index  = permuted_index[  -eval_set_size:                ]

# Split DataFrames
train_df = df.loc[df['uid'].isin(train_user_index)]
val_df   = df.loc[df['uid'].isin(val_user_index)]
test_df  = df.loc[df['uid'].isin(test_user_index)]

# DataFrame to dict => {uid : list of sid's}
train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))

user_row = []
for user, useritem in enumerate(user2items): #removed .values()
    for _ in range(len(useritem)):
        user_row.append(user)

# Column indices for sparse matrix
item_col = []
for useritem in user2items: #removed .value()
    item_col.extend(useritem)

# Construct sparse matrix
assert len(user_row) == len(item_col)
train_data = sps.csr_matrix((np.ones(len(user_row)), (user_row, item_col)), 
                                dtype='float64', shape=(len(user2items), n_items))

# Convert to torch tensor
self.data = torch.FloatTensor(sparse_data.toarray())

#class AEEvalDataset(torch.utils.data.Dataset):
#    def __init__(self, user2items, item_count):
        # Split each user's items to input and label s.t. the two are disjoint
        # Both are lists of np.ndarrays

def split_input_label_proportion(data, label_prop=0.2):
    input_list, label_list = [], []

    for items in data.values():
        items = np.array(items)
        if len(items) * label_prop >= 1:
            # ith item => "chosen for label" if choose_as_label[i] is True else "chosen for input"
            choose_as_label = np.zeros(len(items), dtype='bool')
            chosen_index = np.random.choice(len(items), size=int(label_prop * len(items)), replace=False).astype('int64')
            choose_as_label[chosen_index] = True
            input_list.append(items[np.logical_not(choose_as_label)])
            label_list.append(items[choose_as_label])
        else:
            input_list.append(items)
            label_list.append(np.array([]))

    return input_list, label_list

def init(user2items, item_count):
    input_list, label_list = split_input_label_proportion(user2items)
    
    # Row indices for sparse matrix
    input_user_row, label_user_row = [], []
    for user, input_items in enumerate(input_list):
        for _ in range(len(input_items)):
            input_user_row.append(user)
    for user, label_items in enumerate(label_list):
        for _ in range(len(label_items)):
            label_user_row.append(user)
    input_user_row, label_user_row = np.array(input_user_row), np.array(label_user_row)
    
    # Column indices for sparse matrix
    input_item_col = np.hstack(input_list)
    label_item_col = np.hstack(label_list)
    
    # Construct sparse matrix
    sparse_input = sps.csr_matrix((np.ones(len(input_user_row)), (input_user_row, input_item_col)),
                                    dtype='float64', shape=(len(input_list), item_count))
    sparse_label = sps.csr_matrix((np.ones(len(label_user_row)), (label_user_row, label_item_col)),
                                    dtype='float64', shape=(len(label_list), item_count))
    
    # Convert to torch tensor
    #self.input_data = torch.FloatTensor(sparse_input.toarray())
    #self.label_data = torch.FloatTensor(sparse_label.toarray())
    return sparse_input, sparse_label


new = init(val, n_items)
"""



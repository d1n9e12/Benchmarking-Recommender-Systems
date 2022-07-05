# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:46:43 2021

@author: Daniel
"""

import pandas as pd
import numpy as np
#import bottleneck as bn
import torch

#BEYOND ACCURACY METRICS
#WITH COURTESY FROM https://github.com/microsoft/recommenders/blob/main/recommenders/evaluation/python_evaluation.py
#Only few times where the likes of "col_item" and "col_user" are replaced by "sid" and "uid"
#in the novelty metrics the order of train_df and reco_df has been changed for easier use 
def _get_pairwise_items(df):
    df_user_i1 = df[["uid", "sid"]]
    df_user_i1.columns = ["uid", "i1"]
    df_user_i2 = df[["uid", "sid"]]
    df_user_i2.columns = ["uid", "i2"]
    
    df_user_i1_i2 = pd.merge(df_user_i1, df_user_i2, how = "inner", on = "uid")
    df_pairwise_items = df_user_i1_i2[(df_user_i1_i2["i1"] <= df_user_i1_i2["i2"])].reset_index(drop = True)
    #in the original code  the three end columns are selected by ["uid", "i1", "i2"], this is not necessary here
    return df_pairwise_items

def _get_cooccurrence_similarity(df):
    pairs = _get_pairwise_items(df)
    #pairs = df_pairwise_items
    #train_df = train_data_
    pairs_count = pd.DataFrame({"count": pairs.groupby(["i1", "i2"]).size()}).reset_index()
    item_count = pd.DataFrame({"count": df.groupby(["sid"]).size()}).reset_index()
    item_count["item_sqrt_count"] = item_count["count"] ** 0.5
    item_co_occur = pairs_count.merge(item_count[["sid", "item_sqrt_count"]], left_on = ["i1"], right_on = ["sid"],).drop(columns = ["sid"])
    item_co_occur.columns = ["i1", "i2", "count", "i1_sqrt_count"]
    
    item_co_occur = item_co_occur.merge(item_count[["sid", "item_sqrt_count"]], left_on = ["i2"], right_on = ["sid"],).drop(columns = ["sid"])
    item_co_occur.columns = ["i1", "i2", "count", "i1_sqrt_count", "i2_sqrt_count"]
    
    item_co_occur["sim"] = item_co_occur["count"] / (item_co_occur["i1_sqrt_count"] * item_co_occur["i2_sqrt_count"])
    df_cosine_similarity = (item_co_occur[["i1", "i2", "sim"]]).sort_values(["i1", "i2"]).reset_index(drop = True)
    
    return df_cosine_similarity

#def _get_item_feature_similarity() #todo, research this

def _get_cosine_similarity(df):
    return _get_cooccurrence_similarity(df)
    
def _get_intralist_similarity(reco_df, train_df):
    pairs = _get_pairwise_items(reco_df)
    similarity_df = _get_cosine_similarity(train_df)
    #similarity_df = df_cosine_similarity
    item_pair_sim = pairs.merge(similarity_df, on = ["i1", "i2"], how = "left")
    item_pair_sim["sim"].fillna(0, inplace = True)
    item_pair_sim = item_pair_sim.loc[item_pair_sim["i1"] != item_pair_sim["i2"]].reset_index(drop = True)
    
    df_intralist_similarity = (item_pair_sim.groupby(["uid"]).agg({"sim": "mean"}).reset_index())
    df_intralist_similarity.columns = ["uid", "avg_i1_sim"]
    
    return df_intralist_similarity

def user_diversity(reco_df, train_df):
    df_intralist_similarity = _get_intralist_similarity(reco_df, train_df)
    df_user_diversity = df_intralist_similarity
    df_user_diversity["user_diversity"] = 1 - df_user_diversity["avg_i1_sim"]
    df_user_diversity = (df_user_diversity[["uid", "user_diversity"]].sort_values("uid").reset_index(drop = True))    
    return df_user_diversity

def diversity(reco_df, train_df):
    df_user_diversity = user_diversity(reco_df, train_df)
    avg_diversity = df_user_diversity.agg({"user_diversity": "mean"})[0]
    return avg_diversity
    
    
def historical_item_novelty(reco_df, train_df): #reco_df not needed in this step
    n_records = train_df.shape[0]
    item_count = pd.DataFrame({"count": train_df.groupby(["sid"]).size()}).reset_index()
    item_count["item_novelty"] =  -np.log2(item_count["count"] / n_records)
    df_item_novelty = (item_count[["sid", "item_novelty"]].sort_values("sid").reset_index(drop=True)) #the sid value were already sorted

    return df_item_novelty
    
def novelty(reco_df, train_df):
    df_item_novelty = historical_item_novelty(reco_df, train_df)
    n_recommendations = reco_df.shape[0]
    reco_item_count = pd.DataFrame({"count": reco_df.groupby(["sid"]).size()}).reset_index()
    reco_item_novelty = reco_item_count.merge(df_item_novelty, on="sid")
    reco_item_novelty["product"] = (reco_item_novelty["count"] * reco_item_novelty["item_novelty"])
    avg_novelty = reco_item_novelty.agg({"product": "sum"})[0] / n_recommendations

    return avg_novelty    
    
def user_item_serendipity(reco_df, train_df):
    reco_df["relevance"] = 1.0 #added in wrong place
    df_cosine_similarity = _get_cosine_similarity(train_df)
    reco_user_item = reco_df[["uid", "sid"]]
    reco_user_item["reco_item_tmp"] = reco_user_item["sid"]

    train_user_item = train_df[["uid", "sid"]]
    train_user_item.columns = ["uid", "train_item_tmp"]

    reco_train_user_item = reco_user_item.merge(train_user_item, on=["uid"])
    reco_train_user_item["i1"] = reco_train_user_item[
        ["reco_item_tmp", "train_item_tmp"]
    ].min(axis=1)
    reco_train_user_item["i2"] = reco_train_user_item[
        ["reco_item_tmp", "train_item_tmp"]
    ].max(axis=1)

    reco_train_user_item_sim = reco_train_user_item.merge(
        df_cosine_similarity, on=["i1", "i2"], how="left"
    )
    reco_train_user_item_sim["sim"].fillna(0, inplace=True)

    reco_user_item_avg_sim = (
        reco_train_user_item_sim.groupby(["uid", "sid"])
        .agg({"sim": "mean"})
        .reset_index()
    )
    reco_user_item_avg_sim.columns = [
        "uid",
        "sid",
        "avg_item2interactedHistory_sim",
    ]

    df_user_item_serendipity = reco_user_item_avg_sim.merge(
        reco_df, on=["uid", "sid"]
    )
    df_user_item_serendipity["relevance"] = 1.
    df_user_item_serendipity["user_item_serendipity"] = (
        1 - df_user_item_serendipity["avg_item2interactedHistory_sim"]
    ) * df_user_item_serendipity["relevance"]
    df_user_item_serendipity = (
        df_user_item_serendipity[["uid", "sid", "user_item_serendipity"]]
        .sort_values(["uid", "sid"])
        .reset_index(drop=True)
    )

    return df_user_item_serendipity
    
def user_serendipity(reco_df, train_df):
    df_user_item_serendipity = user_item_serendipity(reco_df, train_df)

    df_user_serendipity = (
        df_user_item_serendipity.groupby("uid")
        .agg({"user_item_serendipity": "mean"})
        .reset_index()
    )
    df_user_serendipity.columns = ["uid", "user_serendipity"]
    df_user_serendipity = df_user_serendipity.sort_values("uid").reset_index(
        drop=True
    )

    return df_user_serendipity
    
def serendipity(reco_df, train_df):
    df_user_serendipity = user_serendipity(reco_df, train_df)
    avg_serendipity = df_user_serendipity.agg({"user_serendipity": "mean"})[0]
    return avg_serendipity    
    
    
    
def catalog_coverage(reco_df, train_df):
    # distinct item count in reco_df
    count_distinct_item_reco = reco_df["sid"].nunique()
    # distinct item count in train_df
    count_distinct_item_train = train_df["sid"].nunique()

    # catalog coverage
    c_coverage = count_distinct_item_reco / count_distinct_item_train
    return c_coverage

def distributional_coverage(
    reco_df, train_df):
    # In reco_df, how  many times each col_item is being recommended
    df_itemcnt_reco = pd.DataFrame(
        {"count": reco_df.groupby(["sid"]).size()}
    ).reset_index()

    # the number of total recommendations
    count_row_reco = reco_df.shape[0]

    df_entropy = df_itemcnt_reco
    df_entropy["p(i)"] = df_entropy["count"] / count_row_reco
    df_entropy["entropy(i)"] = df_entropy["p(i)"] * np.log2(df_entropy["p(i)"])

    d_coverage = -df_entropy.agg({"entropy(i)": "sum"})[0]

    return d_coverage
########################
#BINARY ACCURAY METRICS
########################
#These metrics are based on the NDCG_binary_at_k_batch and Recall_binary_at_k_batch from
# https://github.com/younggyoseo/vae-cf-pytorch/blob/master/metric.py
#or the original implementation https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
#for both metrics, the names have been changed to easier ones.
#The others metrics  are a combinations of the top chunks of the recall metric with the 
#metric implementations of https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/blob/master/Evaluation/metrics.py
#where the "is_relevant" from MaurizioFD equals the (np.logical_and(X_true_binary, X_pred_binary) step below 
def ndcg_binary(X_pred, heldout_batch, k=100):
    '''
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1) 
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return np.mean(DCG / IDCG)


def recall_binary(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0).toarray() #for everything except BERT4REC, there's it's .numpy()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return np.mean(recall)

def recall_sequence(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(axis=1) #this is the idx in the original
    cut = rank[:, :k] # comparable to X_pred_binary
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def Beyond_accuracy_binary(X_pred, heldout_batch, k = 100): #if one value, pass user as list
    #Create dataframe for the predictions
    batch_users = X_pred.shape[0]
    idx = (-X_pred).argsort(axis = 1) #equivalent to torch.argsort????
    idx_k = idx[: , :k]
    users_k = []
    for value in range(batch_users):
        users_k.extend([value] * k)
    idx_k_unchained = []
    for idx in idx_k:
        idx_k_unchained.extend(idx)

    predictions_df = pd.DataFrame({"uid" : users_k, "sid" : idx_k_unchained})
    
    #Create dataframe for the heldout data
    user_item_dict = {user : heldout_batch[user].indices for user in range(batch_users)}
    user_times_occurences, item_times_occurences = [], []
    for user, items in user_item_dict.items():
        for item in items:
            user_times_occurences.append(user)
            item_times_occurences.append(item)

    heldout_df = pd.DataFrame({"uid": user_times_occurences, "sid" : item_times_occurences}) #could convert to list that item array
 
    return predictions_df, heldout_df


def Beyond_accuracy_prep_sas(X_pred, heldout_batch, k = 100): #if one value, pass user as list
    #Create dataframe for the predictions

    batch_users = X_pred.shape[0]
    scores = X_pred.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    users_k = []
    for value in range(batch_users):
        users_k.extend([value] * k)
    cut_k_unchained = []
    for idx in cut.detach().numpy():
        cut_k_unchained.extend(idx)

    predictions_df = pd.DataFrame({"uid" : users_k, "sid" : cut_k_unchained})
    
    #Create dataframe for the heldout data
    #candidates = candidates #.detach().numpy()
    users_ho = []
    for value in range(batch_users):
        users_ho.extend([value] * len(heldout_batch))

    heldout_df = pd.DataFrame({"uid": users_ho, "sid" : heldout_batch}) #could convert to list that item array
 
    return predictions_df, heldout_df


#The following code has been taken from https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch and
# #with courtesy from https://github.com/Furyton/Recommender-Baseline-Model/blob/main/NerualNetwork/bert4rec%26sas4rec/trainers/utils.py
#and adapted accordingly to https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi/blob/master/Evaluation/metrics.py

#The recall and ndcg are taken from jaywonchung and mrr is taken from Furyton. However, 
#it is the arhr which is erronuously called mrr.
#precision, hitrate, mrr, map, (mar) are changed accordingly to the metrics implemented by Maurizio Dacrema
#in this case the hits tensor from the first 2 authors equals the is_relevant from the last one.
##################################
##################################
#In order to make a transition to the metrics from the microsoft recommender system library,
#the item list and tensors have to be changed to a pd.Dataframe
def Beyond_accuracy_prep_bert(X_pred, total_train, k = 100): #if one value, pass user as list
    #Create dataframe for the predictions
    batch_users = X_pred.shape[0]
    scores = X_pred.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    users_k = []
    for value in range(batch_users):
        users_k.extend([value] * k)
    cut_k_unchained = []
    for idx in cut.detach().numpy():
        cut_k_unchained.extend(idx)

    predictions_df = pd.DataFrame({"uid" : users_k, "sid" : cut_k_unchained})
    
    #Create dataframe for the heldout data
    total_train = total_train.detach().numpy()
    user_item_dict = {user : total_train[user] for user in range(batch_users)}
    user_times_occurences, item_times_occurences = [], []
    for user, items in user_item_dict.items():
        for item in items:
            if item != 0:
                user_times_occurences.append(user)
                item_times_occurences.append(item)

    heldout_df = pd.DataFrame({"uid": user_times_occurences, "sid" : item_times_occurences}) #could convert to list that item array
 
    return predictions_df, heldout_df

###############################################################
#The following metrics have been specifically designed to evaluate sequential recommenders
"""
def precision_sequence(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(axis=1) #this is the idx in the original
    cut = rank[:, :k] # comparable to X_pred_binary
    hits = labels.gather(1, cut)
    return (hits.sum(1).float() / torch.Tensor([k])).mean().item() #that .to(hit.device) removed

#scores = model2(seqs)  # B x T x V
#scores = scores[:, -1, :]  # B x V
#scores = scores.gather(1, candidates)  # B x C
def recall_sequence(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(axis=1) #this is the idx in the original
    cut = rank[:, :k] # comparable to X_pred_binary
    hits = labels.gather(1, cut)
    return (hits.sum(1).float() / torch.min(torch.Tensor([k]), labels.sum(1).float())).mean().item()
"""
"""
def hits_sequence(scores, labels, k):
    #is_relevant = np.logical_and(X_true_binary, X_pred_binary)
    #all(i == j for i,j in zip(is_relevant.sum(1), hit.sum(1)))
    scores = scores
    labels = labels
    rank = (-scores).argsort(axis=1) #this is the idx in the original
    cut = rank[:, :k] # comparable to X_pred_binary
    hits = labels.gather(1, cut)
    #np.mean([np.any(relevant) for relevant in is_relevant ]) == np.mean([np.any(hit_) for hit_ in hit.numpy() ])
    return np.mean([np.any(hit_) for hit_ in hits.numpy() ])
"""
def precision_sequence(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(axis=1) #this is the idx in the original
    cut = rank[:, :k] # comparable to X_pred_binary
    hits = labels.gather(1, cut)
    return (hits.sum(1).float() / torch.min(torch.Tensor([k]), labels.sum(1).float())).mean().item()


#scores = model2(seqs)  # B x T x V
#scores = scores[:, -1, :]  # B x V
#scores = scores.gather(1, candidates)  # B x C
def recall_sequence(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(axis=1) #this is the idx in the original
    cut = rank[:, :k] # comparable to X_pred_binary
    hits = labels.gather(1, cut)
    return (hits.sum(1).float() / labels.sum(1).float()).mean().item()

def f1_sequence(scores, labels, k):
    prec = precision_sequence(scores, labels, k)
    rec = recall_sequence(scores, labels, k)
    return 2 * prec * rec / (prec + rec)

def hits_sequence(scores, labels, k):
    #is_relevant = np.logical_and(X_true_binary, X_pred_binary)
    #all(i == j for i,j in zip(is_relevant.sum(1), hit.sum(1)))
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(axis=1) #this is the idx in the original
    cut = rank[:, :k] # comparable to X_pred_binary
    hits = labels.gather(1, cut)
    return np.mean(np.any(hits.numpy(), axis = 1))

def ndcg_sequence(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()

def mrr_sequence(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels_float.gather(1, cut).numpy().astype(bool)

    rank_l = []
    for user in range(scores.shape[0]):
        ranks = np.arange(1, hits.shape[1] + 1)[hits[0]] #that second dimension had to go
        
        if ranks.shape[0] > 0:
            rank_l.append(1. / ranks[0])
        else:
            rank_l.append(0.)  

    return np.mean(rank_l)

def arhr_sequence(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank

    cut = cut[:, :k]
    hits = labels_float.gather(1, cut)

    position_arhr = torch.arange(1, k + 1)
    weights_arhr = 1 / position_arhr.float()
    arhr = (hits * weights_arhr).sum(1)
    arhr = arhr.mean().item()

    return arhr
"""
def map_sequence(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    p_at_k = hits.numpy() * np.cumsum(hits.numpy(), axis = 1, dtype=np.float64) / (1 + np.arange(hits.numpy().shape[1])) #look for explanation beneath
    a_p = np.sum(p_at_k, axis = 1) / p_at_k.shape[1] #batch_size results in 1, doesn't seem useful, take shape[1] because it's about the length of the list, but here it's a 2dim array
    return np.mean(a_p)
"""
def map_sequence(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    p_at_k = hits * torch.cumsum(hits, dim = 1) / (1 + torch.arange(k)) #look for explanation beneath
    a_p = torch.sum(p_at_k, dim = 1) / torch.min(torch.Tensor([k]), labels.sum(1).float()) #batch_size results in 1, doesn't seem useful, take shape[1] because it's about the length of the list, but here it's a 2dim array
    return a_p.mean().item()

def mar_sequence(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    r_at_k = hits * torch.cumsum(hits, dim = 1) / (1 + torch.arange(k)) #look for explanation beneath
    a_r = torch.sum(r_at_k, dim = 1) / labels.sum(1).float()  #batch_size results in 1, doesn't seem useful, take shape[1] because it's about the length of the list, but here it's a 2dim array
    return a_r.mean().item()

#The loop used in the very end
metrics_n = ["PREC@%d", "REC@%d", "F1@%d", "HIT@%d", "NDCG@%d", "MRR@%d", "ARHR@%d", "MAP@%d", "MAR@%d",
             "DIV@%d", "NOV@%d", "SER@%d", "CC@%d", "DC@%d"]
ks = [1, 5, 10, 20, 50, 100]
metrics = {metric_n % k: [] for metric_n in metrics_n for k in ks}

#original loop taken from Furyton, slightly adapted

def accuracy_sequences(recon_batch, labels, ks = [1, 5, 10, 20, 50, 100]):
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
            



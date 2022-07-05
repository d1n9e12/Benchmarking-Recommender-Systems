# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:19:42 2022

@author: Daniel
"""
import ray    
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune import CLIReporter
from functools import partial
def train_ease(config, checkpoint_dir = None):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
    ease = EASE_R_Recommender(train_data)
    ease.fit(topK = config["topK"], l2_norm = config["l2_norm"],
             normalize_matrix = config["normalize_matrix"])
    
    prec100_l, rec100_l, f100_l, hit100_l = [], [], [], []
    ndcg100_l, mrr100_l, arhr100_l, map100_l, mar100_l = [], [], [], [], []
    div100_l, nov100_l, ser100_l, cc100_l, dc100_l = [], [], [], [], []
    
    users = list(range(train_data.shape[0]))
    for user in users:
        print(user)
        recon_batch = ease._compute_item_score(user)
        recon_batch[train_data[user].nonzero()] = -np.inf
        recon_batch = torch.FloatTensor(recon_batch)
        labels = labels_val[user].unsqueeze(0)

        prec100_l.append(precision_sequence(recon_batch, labels, 100))
        rec100_l.append(recall_sequence(recon_batch, labels, 100))
        f100_l.append(f1_sequence(recon_batch, labels, 100))
        hit100_l.append(hits_sequence(recon_batch, labels, 100))
        ndcg100_l.append(ndcg_sequence(recon_batch, labels, 100))
        mrr100_l.append(mrr_sequence(recon_batch, labels, 100))
        arhr100_l.append(arhr_sequence(recon_batch, labels, 100))
        map100_l.append(map_sequence(recon_batch, labels, 100))
        mar100_l.append(mar_sequence(recon_batch, labels, 100))
        beyond = Beyond_accuracy_binary(recon_batch.numpy(), train_data[user], 100)
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
    "topK" : tune.choice( list(range(300))), "l2_norm" : tune.choice(list(range(1, 100))),
    "normalize_matrix" : tune.choice([True, False])}        

#config = {"topK": 100, "l2_norm": tune.loguniform(1e-20, 1e-1), "normalize_matrix": True}

scheduler = ASHAScheduler(
        metric= "ndcg100",
        mode="max",) 
 
reporter = CLIReporter(metric_columns=["prec100", "rec100", "f100", "hit100",
                                       "ndcg100", "mrr100", "arhr100", "map100", "mar100"
                                       "div100", "nov100", "ser100", "cc100", "dc100"])
bayesopt = SkOptSearch(metric = "ndcg100", mode = "max")                                  

result_ease = tune.run(
    partial(train_ease),
    #resources_per_trial={"cpu": 8}, #, "gpu": gpus_per_trial},
    config=config,
    num_samples= 20,
    scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end= False,
    fail_fast = "raise",
    search_alg = bayesopt
    )



best_trial_ease = result_ease.get_best_trial("ndcg100", "max", "last")
print("Best trial config: {}".format(best_trial_ease.config))

print("Best trial final validation precision: {}".format(best_trial_ease.last_result["prec100"]))
print("Best trial final validation recall: {}".format(best_trial_ease.last_result["rec100"]))
print("Best trial final validation f1: {}".format(best_trial_ease.last_result["f100"]))
print("Best trial final validation hits: {}".format(best_trial_ease.last_result["hit100"]))

print("Best trial final validation ndcg: {}".format(best_trial_ease.last_result["ndcg100"]))
print("Best trial final validation mrr: {}".format(best_trial_ease.last_result["mrr100"]))
print("Best trial final validation arhr: {}".format(best_trial_ease.last_result["arhr100"]))
print("Best trial final validation map: {}".format(best_trial_ease.last_result["map100"]))
print("Best trial final validation mar: {}".format(best_trial_rp3beta.last_result["mar100"]))


print("Best trial final validation novelty: {}".format(best_trial_ease.last_result["nov100"]))
print("Best trial final validation diversity: {}".format(best_trial_ease.last_result["div100"]))
print("Best trial final validation serendipity: {}".format(best_trial_ease.last_result["ser100"]))
print("Best trial final validation catalog coverage: {}".format(best_trial_ease.last_result["cc100"]))
print("Best trial final validation distributional coverage: {}".format(best_trial_ease.last_result["dc100"]))

best_ease = EASE_R_Recommender(train_data)
best_ease.fit(topK = best_trial_ease.config["topK"], l2_norm = best_trial_ease.config["l2_norm"],
              normalize_matrix = best_trial_ease.config["normalize_matrix"])


metrics_n = ["PREC@%d", "REC@%d", "F1@%d", "HIT@%d", "NDCG@%d", "MRR@%d", "ARHR@%d", "MAP@%d", "MAR@%d",
             "DIV@%d", "NOV@%d", "SER@%d", "CC@%d", "DC@%d"]
ks = [1, 5, 10, 20, 50, 100]
metrics = {metric_n % k: [] for metric_n in metrics_n for k in ks}
users = list(range(train_data.shape[0]))

for user in users:
    recon_batch = best_ease._compute_item_score(user)
    recon_batch[train_data[user].nonzero()] = -np.inf
    recon_batch = torch.FloatTensor(recon_batch)
    labels = labels_test[user].unsqueeze(0)

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
        
        a_r = torch.sum(p_at_k, dim = 1) / labels.sum(1).float() 
        metrics['MAR@%d' % k].append(a_r.mean().item())
             
        beyond = Beyond_accuracy_binary(recon_batch.detach().numpy(), train_data[e_idxlist[start_idx: end_idx]] , k) #candidates_test[user].numpy()
        metrics['DIV@%d' % k].append(diversity(beyond[0], beyond[1]))
        metrics['NOV@%d' % k].append(novelty(beyond[0], beyond[1]))
        metrics['SER@%d' % k].append(serendipity(beyond[0], beyond[1]))
        metrics['CC@%d' % k].append(catalog_coverage(beyond[0], beyond[1]))
        metrics['DC@%d' % k].append(distributional_coverage(beyond[0], beyond[1]))
            
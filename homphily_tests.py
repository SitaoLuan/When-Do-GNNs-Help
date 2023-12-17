import argparse
import os

import numpy as np
import torch
import torch.nn.functional as f
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from utils.homophily_metrics import random_disassortative_splits, classifier_based_performance_metric, similarity, \
    adjusted_homo, \
    label_informativeness, node_homophily, our_measure, edge_homophily, generalized_edge_homophily
from utils.util_funcs import row_normalized_adjacency, sys_normalized_adjacency, full_load_data_large, normalize_tensor, \
    sparse_mx_to_torch_sparse_tensor

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
device = torch.device(device)

ifsum = 1
num_exp = 10

BASE_CLASSIFIERS = ['kernel_reg0', 'kernel_reg1', 'gnb']
SMALL_DATASETS = ['cornell', 'wisconsin', 'texas', 'film', 'chameleon', 'squirrel', 'cora', 'citeseer', 'pubmed']
LARGE_DATASETS = ['deezer-europe', 'Penn94', 'arxiv-year', "genius", "twitch-gamer", 'pokec', 'snap-patents']
DATASETS = SMALL_DATASETS + LARGE_DATASETS
METRIC_LIST = {
    "node_homo": lambda adj, labels: node_homophily(adj, labels),
    "edge_homo": lambda adj, labels: edge_homophily(adj, labels),
    "class_homo": lambda adj, labels: our_measure(adj, labels),
    "node_hom_generalized": lambda adj, features, labels: generalized_edge_homophily(adj, features, labels),
    "agg_homo_soft": lambda x: np.mean(x),
    "agg_homo_hard": lambda x: np.mean(x),
    "adj_homo": lambda adj, labels: adjusted_homo(adj, labels),
    "label_info": lambda adj, labels: label_informativeness(adj, labels),
    "kernel_reg0_based_homo": lambda *args, **kwargs: classifier_based_performance_metric(*args, **kwargs),
    "kernel_reg1_based_homo": lambda *args, **kwargs: classifier_based_performance_metric(*args, **kwargs),
    "gnb_based_homo": lambda *args, **kwargs: classifier_based_performance_metric(*args, **kwargs)
}

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--dataset_name', type=str, required=True, choices=DATASETS,
                    help=f"The data set name, please select from the following list: \n"
                         f"{DATASETS}")
parser.add_argument('--symmetric', type=float, default=0,
                    help='1 for symmetric renormalized adj, 0 for random walk renormalized adj')
parser.add_argument('--sample_max', type=float, default=500, help='maxinum number of samples used in gntk')
parser.add_argument('--base_classifier', type=str, default='kernel_reg1', choices=BASE_CLASSIFIERS,
                    help='The classifier used for performance metric(kernel_reg1, kernel_reg0, svm_linear, svm_rbf, svm_poly, gnb)')
parser.add_argument('--homophily_metric', required=True, choices=list(METRIC_LIST.keys()),
                    help="The metric to measure homophily, please select from the following list: \n"
                         "{node_homo (node homophily), \n"
                         " edge_homo (edge homophily), \n"
                         " class_homo (class homophily), \n"
                         " node_hom_generalized (generalized node homophily), \n"
                         " agg_homo_soft (aggreation homophily with soft LAS), \n"
                         " agg_homo_hard (aggreation homophily with hard LAS), \n"
                         " adj_homo (adjusted homophily), \n"
                         " label_info (label informativeness), \n"
                         " kernel_reg0_based_homo (kernel based homophily with reg0), \n"
                         " kernel_reg1_based_homo (kernel based homophily with reg1), \n"
                         " gnb_based_homo (gnd-based homophily)}")

args = parser.parse_args()
dataset_name = args.dataset_name
homophily_metric = args.homophily_metric
homophily_lvl = -1

if dataset_name in SMALL_DATASETS:
    adj_low_unnormalized, features, labels = full_load_data_large(dataset_name)
    features = normalize_tensor(features).to(device)
    nnodes = (labels.shape[0])

    adj = normalize_tensor(torch.eye(nnodes) + adj_low_unnormalized.to_dense(),
                           symmetric=args.symmetric).to(device)
    adj = adj.to_sparse().to(device)
    labels = labels.to(device)
else:
    adj_low_unnormalized, features, labels = full_load_data_large(dataset_name)
    nnodes = (labels.shape[0])
    adj_low_pt = 'acmgcn_features/' + dataset_name + '_adj_low.pt'
    adj_high_pt = 'acmgcn_features/' + dataset_name + '_adj_high.pt'

    features = f.normalize(features, p=1, dim=1)
    if os.path.exists(adj_low_pt) and os.path.exists(adj_high_pt):
        adj = torch.load(adj_low_pt)

    else:
        adj = to_scipy_sparse_matrix(adj_low_unnormalized.coalesce().indices())
    if args.symmetric == 1:
        adj = sys_normalized_adjacency(adj)
    else:
        adj = row_normalized_adjacency(adj)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    torch.save(adj, adj_low_pt)
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    if labels.dim() > 1:
        labels = labels.flatten()

if homophily_metric in ("node_homo", "class_homo", "label_info", "adj_homo"):
    homophily_lvl = METRIC_LIST[homophily_metric](adj, labels)
elif homophily_metric == "edge_homo":
    labels = torch.eye(labels.max() + 1)[labels]
    homophily_lvl = METRIC_LIST[homophily_metric](adj, labels)
elif homophily_metric == "node_hom_generalized":
    homophily_lvl = METRIC_LIST[homophily_metric](adj, features, labels)
elif homophily_metric in ("agg_homo_soft", "agg_homo_hard"):
    adj, features, labels = full_load_data_large(dataset_name)
    las = np.zeros(10)
    is_hard = 1 if homophily_metric.partition("agg_homo_")[-1] == "hard" else None
    # Compute aggregation Homophily
    num_sample = 10000
    label_onehot = torch.eye(labels.max() + 1)[labels]
    for i in range(10):
        if nnodes >= num_sample:
            idx_train, _, _ = random_disassortative_splits(labels, labels.max() + 1, num_sample / nnodes)
        else:
            idx_train = None
        las[i] = 2 * similarity(label_onehot, adj, label_onehot, hard=is_hard, LP=1, idx_train=idx_train) - 1
    homophily_lvl = METRIC_LIST[homophily_metric](las)
elif homophily_metric in ("kernel_reg0_based_homo", "kernel_reg1_based_homo", "gnb_based_homo"):
    base_classifier = homophily_metric.partition("_based")[0]
    adj, features, labels = full_load_data_large(dataset_name)
    homophily_lvl = METRIC_LIST[homophily_metric](features, adj, labels, args.sample_max,
                                                  base_classifier=base_classifier, epochs=100)

print(f"The Homophily level of given dataset {dataset_name} is {homophily_lvl} using metric {homophily_metric}")

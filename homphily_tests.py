import argparse
import os

import numpy as np
import torch
import torch.nn.functional as f
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from homophily2 import random_disassortative_splits, classifier_based_performance_metric, similarity, adjusted_homo, \
    label_informativeness, node_homophily, our_measure, edge_homophily, generalized_edge_homophily
from utils_large import row_normalized_adjacency, sys_normalized_adjacency, full_load_data_large, normalize_tensor, \
    sparse_mx_to_torch_sparse_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--dataset_name', type=str,
                    help='Dataset name.', default='chameleon')
parser.add_argument('--symmetric', type=float, default=0,
                    help='1 for symmetric renormalized adj, 0 for random walk renormalized adj')
parser.add_argument('--sample_max', type=float, default=500, help='maxinum number of samples used in gntk')
parser.add_argument('--base_classifier', type=str, default='kernel_reg1',
                    help='The classifier used for performance metric(kernel_reg1, kernel_reg0, svm_linear, svm_rbf, svm_poly, gnb)')

args = parser.parse_args()
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
device = torch.device(device)

ifsum = 1
num_exp = 10
base_classifier = ['kernel_reg0', 'kernel_reg1', 'gnb']
small_datasets = ['cornell', 'wisconsin', 'texas', 'film', 'chameleon', 'squirrel', 'cora', 'citeseer', 'pubmed']
large_datasets = ['deezer-europe', 'Penn94', 'arxiv-year', "genius", "twitch-gamer", 'pokec', 'snap-patents']
all_datasets = small_datasets + large_datasets

for dataset_name in all_datasets:
    if dataset_name in small_datasets:
        adj_low_unnormalized, features, labels = full_load_data_large(dataset_name)

        features = normalize_tensor(features).to(device)

        nnodes = (labels.shape[0])

        adj = normalize_tensor(torch.eye(nnodes) + adj_low_unnormalized.to_dense(),
                               symmetric=args.symmetric).to(device)
        adj = adj.to_sparse().to(device)
        labels = labels.to(device)

    elif dataset_name in large_datasets:
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
    adj, features, labels = full_load_data_large(dataset_name)
    soft_las = np.zeros(10)
    hard_las = np.zeros(10)

    # Compute aggregation Homophily
    num_sample = 10000
    label_onehot = torch.eye(labels.max() + 1)[labels]
    for i in range(10):
        if nnodes >= num_sample:
            idx_train, _, _ = random_disassortative_splits(labels, labels.max() + 1, num_sample / nnodes)
        else:
            idx_train = None

        soft_las[i] = 2 * similarity(label_onehot, adj, label_onehot, hard=None, LP=1, idx_train=idx_train) - 1
        hard_las[i] = 2 * similarity(label_onehot, adj, label_onehot, hard=1, LP=1, idx_train=idx_train) - 1

    kernel_reg0 = classifier_based_performance_metric(features, adj, labels, args.sample_max,
                                                      base_classifier='kernel_reg0', epochs=100)
    kernel_reg1 = classifier_based_performance_metric(features, adj, labels, args.sample_max,
                                                      base_classifier='kernel_reg1', epochs=100)
    gnb = classifier_based_performance_metric(features, adj, labels, args.sample_max, base_classifier='gnb', epochs=100)
    adj_homo = adjusted_homo(adj, labels)
    LI = label_informativeness(adj, labels)
    print(dataset_name, ' Node Homo: ', node_homophily(adj.to(device), labels.to(device)), 'Edge Homo',
          edge_homophily(adj, torch.eye(labels.max() + 1)[labels]), "Class Homo: ", our_measure(adj, labels),
          "Generalized Edge Homo: ", generalized_edge_homophily(adj, features, labels),
          'Aggregation Homo (sofs las): ', np.mean(soft_las), 'Aggregation Homo (hard las): ', 'Adjusted Homo: ',
          adj_homo, 'Label Informativeness: ', LI, np.mean(hard_las), "kernel_reg0-based Homo: ", kernel_reg0,
          "kernel_reg1-based Homo: ", kernel_reg1, "gnb-based Homo: ", gnb)

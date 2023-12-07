from __future__ import division
from __future__ import print_function
from pathlib import Path
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
# import torch.nn.functional as f
import torch.nn.functional as f
import torch.optim as optim
import matplotlib
import scipy
# from homophily import *
from homophily2 import *
from utils_large import load_data,load_fixed_splits,row_normalized_adjacency,sys_normalized_adjacency,get_adj_high, accuracy, full_load_data_large, data_split, normalize, normalize_tensor, train, normalize_adj,sparse_mx_to_torch_sparse_tensor, dataset_edge_balance, random_disassortative_splits, eval_rocauc,eval_acc, load_fixed_splits, evaluate, WGCN_split, SuperGAT_split, Optimal_hyperparameters,Optimal_hyperparameters_sym
# from Homophily_stats import similarity
from torch_geometric.utils.convert import to_scipy_sparse_matrix
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--dataset_name', type=str,
                    help='Dataset name.', default = 'chameleon') #wisconsin?
parser.add_argument('--symmetric', type=float, default=0, help='1 for symmetric renormalized adj, 0 for random walk renormalized adj')
parser.add_argument('--sample_max', type=float, default=500, help='maxinum number of samples used in gntk')
parser.add_argument('--base_classifier', type=str, default='kernel_reg1', help='The classifier used for performance metric(kernel_reg1, kernel_reg0, svm_linear, svm_rbf, svm_poly, gnb)')

args = parser.parse_args()
if torch.cuda.is_available():
    device = 'cuda:0' 
else:
    device = 'cpu'
device = torch.device(device)


ifsum = 1
num_exp = 10
base_classifier = ['kernel_reg0', 'kernel_reg1', 'gnb']
small_datasets = ['cornell','wisconsin','texas','film','chameleon','squirrel','cora','citeseer','pubmed']
large_datasets = ['deezer-europe','Penn94', 'arxiv-year',"genius","twitch-gamer", 'pokec','snap-patents']
all_datasets  = small_datasets+large_datasets 

for dataset_name in all_datasets:  
    if dataset_name in small_datasets:
        adj_low_unnormalized, features, labels = full_load_data_large(dataset_name)
        
        features = normalize_tensor(features).cpu()#.to(device)
            
        nnodes = (labels.shape[0])

        adj = normalize_tensor(torch.eye(nnodes) + adj_low_unnormalized.to_dense(), symmetric = args.symmetric).cpu()#.to(device)
        adj = adj.to_sparse().cpu()#.to(device)
        labels  = labels.cpu()#.to(device)
        
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
            features = features.cpu()#.to(device)
            adj = adj.cpu()##.to(device)       
            labels = labels.cpu()#.to(device)
    if labels.dim() >1:
        labels = labels.flatten()
    adj, _, features, labels = full_load_data(dataset_name)
    print(dataset_name, similarity_low_identity_ratio(features, adj, torch.eye(labels.max()+1)[labels], idx_train = None))
    soft_las=np.zeros(10)
    hard_las=np.zeros(10)
    #Compute aggregation Homophily
    num_sample =10000
    label_onehot = torch.eye(labels.max()+1)[labels]
    for i in range(10):
        if nnodes >= num_sample:
            idx_train, _, _ = random_sdisassortative_splits(labels, labels.max()+1, num_sample/nnodes)
        else:
            idx_train = None
            
        soft_las[i] = 2*similarity(label_onehot, adj, label_onehot, NTK=None, hard = None, LP = 1, idx_train = idx_train) -1 #torch.mm(adj,adj)
        hard_las[i] = 2*similarity(label_onehot, adj, label_onehot, NTK=None, hard = 1, LP = 1, idx_train  = idx_train) - 1 #torch.mm(adj,adj)
        
    kernel_reg0 = classifier_based_performance_metric(features, adj, labels, args.sample_max, base_classifier = 'kernel_reg0', epochs = 100)
    kernel_reg1 = classifier_based_performance_metric(features, adj, labels, args.sample_max, base_classifier = 'kernel_reg1', epochs = 100)
    gnb = classifier_based_performance_metric(features, adj, labels, args.sample_max, base_classifier = 'gnb', epochs = 100)
    adj_homo = adjusted_homo(adj, labels)
    LI = label_informativeness(adj, labels)
    print(dataset_name, ' Node Homo: ',node_homophily(adj.cpu(), labels.cpu()), 'Edge Homo', edge_homophily(adj, torch.eye(labels.max()+1)[labels]), "Class Homo: ", our_measure(adj, labels),  "Generalized Edge Homo: ",generalized_edge_homophily(adj, features, labels),
          'Aggregation Homo (sofs las): ', np.mean(soft_las), 'Aggregation Homo (hard las): ', 'Adjusted Homo: ', adj_homo, 'Label Informativeness: ', LI, np.mean(hard_las), "kernel_reg0-based Homo: ", kernel_reg0, "kernel_reg1-based Homo: ", kernel_reg1, "gnb-based Homo: ", gnb)     



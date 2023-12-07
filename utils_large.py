import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
import json
from networkx.readwrite import json_graph
import pdb
import os
import re
import pandas
import random
import torch.nn as nn
import torch as th
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize
from numpy.linalg import matrix_power
from os import path
import dgl
from google_drive_downloader import GoogleDriveDownloader as gdd
import scipy
import scipy.io
import scipy.sparse
import csv
from sklearn.preprocessing import normalize as sk_normalize
import pandas as pd
from torch_geometric.utils import to_undirected, add_self_loops
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from ogb.nodeproppred import NodePropPredDataset
import random 
DATAPATH = path.dirname(path.abspath(__file__)) + '/data/'
small_datasets = ['cornell','wisconsin','texas','film','chameleon','squirrel','cora','citeseer','pubmed']#, 'cornell','wisconsin','texas','film','chameleon','squirrel','cora','citeseer','pubmed', 'CitationFull_dblp', 'Coauthor_CS', 'Coauthor_Physics','Amazon_Computers', 'Amazon_Photo','Flickr'  ['chameleon'] #
large_datasets = ['deezer-europe', 'yelp-chi','Penn94', 'arxiv-year', 'pokec','snap-patents',"genius","twitch-gamer","wiki"]




if torch.cuda.is_available():
    from collections import defaultdict
    import scipy.io
    from sklearn.preprocessing import label_binarize
    
    # from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.utils import add_self_loops, to_dense_adj, contains_self_loops, remove_self_loops, to_dense_adj, to_undirected
    
    from torch_sparse import SparseTensor

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
# 
#     from torch_geometric.datasets import CitationFull, Coauthor, Amazon, Flickr, WikiCS
#     from torch_geometric.utils import to_dense_adj, contains_self_loops, remove_self_loops
    
sys.setrecursionlimit(99999)

def train(model, optimizer, adj_low, adj_high, adj_low_unnormalized,  adj_label, features, labels, idx_train, idx_val, criterion, dataset_name):
    model.train()
    # adj_label_train, adj_label_train_c = torch.zeros(adj_label.size()).to(device), torch.zeros(adj_label.size()).to(device)
    # adj_label_train[idx_train.float().nonzero().squeeze(1),idx_train.float().nonzero()] = adj_label[idx_train,:][:,idx_train]
    # # D= torch.sum(adj_label_train,1)
    
    # adj_label_c = torch.ones(adj_label.size()) - adj_label
    # adj_label_c = adj_label_c - torch.diag(torch.diag(adj_label_c))
    # adj_label_train_c[idx_train.float().nonzero().squeeze(1),idx_train.float().nonzero()] = adj_label_c[idx_train,:][:,idx_train]
    # D_c = torch.sum(adj_label_train_c,1)
    
    
    
    optimizer.zero_grad()
    output = model(features, adj_low, adj_high, adj_low_unnormalized)
    if dataset_name in large_datasets:
        if dataset_name in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
            if labels.shape[1] == 1:
                labels = F.one_hot(labels, labels.max() + 1).squeeze(1)
            else:
                labels = labels
            loss_train = criterion(output[idx_train],  labels.squeeze(1)[idx_train].to(torch.float))
            acc_train = eval_rocauc(labels[idx_train], output[idx_train])
        else:
            output = F.log_softmax(output, dim=1)
            loss_train = criterion(output[idx_train], labels.squeeze(1)[idx_train])
            acc_train = eval_acc(labels[idx_train], output[idx_train])
    else:
        ##reg = torch.mean(torch.sum(torch.mm(output, torch.transpose(output,0,1)) * (adj_label_train - adj_label_train_c),1))
        output = F.log_softmax(output, dim=1)
        loss_train = criterion(output[idx_train], labels[idx_train]) #- 0.01*reg
        # + 1/sum(idx_train) * torch.trace(torch.mm(torch.mm(torch.exp(output).transpose(0,1),D-adj_label_train), torch.exp(output)))
        # - 1/sum(idx_train) * torch.trace(torch.mm(torch.mm(torch.exp(output).transpose(0,1),D_c-adj_label_train_c), torch.exp(output)))
        acc_train = accuracy(labels[idx_train], output[idx_train])
            
    loss_train.backward()
    optimizer.step()
    if dataset_name in large_datasets:
        return  acc_train, loss_train.item()
    else:
        return 100 * acc_train.item(), loss_train.item()


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = (1.0 / rowsum).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, features, labels

def load_torch_geometric_data(dataset, name):
    cur = os.getcwd()

    if dataset in {'WikiCS', 'Flickr'}:
        data = eval(dataset + "(root = '" + cur.replace("\\", "/") + "/torch_geometric_data/" + dataset + "')")
    else:
        data = eval(dataset + "(root = '" + cur.replace("\\", "/") + "/torch_geometric_data/" + dataset + "'," + "name = '" + name + "')")
    # e.g. Coauthor(root='...', name = 'CS')

    edge = data[0].edge_index
    if contains_self_loops(edge):
        edge = remove_self_loops(edge)[0]
        print("Original data contains self-loop, it is now removed")

    adj = to_dense_adj(edge)[0].numpy()

    print("Nodes: %d, edges: %d, features: %d, classes: %d. \n"%(len(adj[0]), len(edge[0])/2, len(data[0].x[0]), len(np.unique(data[0].y))))

    mask = np.transpose(adj) != adj
    col_sum = adj.sum(axis=0)
    print("Check adjacency matrix is sysmetric: %r"%(mask.sum().item() == 0))
    print("Check the number of isolated nodes: %d"%((col_sum == 0).sum().item()))
    print("Node degree Max: %d, Mean: %.4f, SD: %.4f"%(col_sum.max(), col_sum.mean(), col_sum.std()))

    return adj, data[0].x.numpy(), data[0].y.numpy()


def full_load_data_large(dataset_name, sage_data=False):
    #splits_file_path = 'splits/'+dataset_name+'_split_0.6_0.2_'+str(idx)+'.npz'
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels = load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        # adj = nx.DiGraph(adj).to_undirected()
    elif dataset_name in {'CitationFull_dblp', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Computers', 'Amazon_Photo'}:
        dataset, name = dataset_name.split("_")
        adj, features, labels = load_torch_geometric_data(dataset, name)

    elif dataset_name in {'Flickr', 'WikiCS'}:
        adj, features, labels = load_torch_geometric_data(dataset_name, None)
    elif dataset_name in {'Crocodile-5'}:
        adj, features, labels = read_WGCN_crocodile() 
        
    elif dataset_name in {'Crocodile-6'}:
        adj, features, labels = read_SuperGAT_crocodile()
    elif dataset_name == 'deezer-europe':
        dataset = load_deezer_dataset()
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        # _, adj_low, _ = gen_normalized_adjs(dataset)
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        # adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label
        # return adj_low, features, labels
    elif dataset_name == 'yelp-chi':
        dataset = load_yelpchi_dataset()
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        # adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label.unsqueeze(1)
        # _, adj_low, _ = gen_normalized_adjs(dataset)
        # features, labels = th.FloatTensor(preprocess_features(dataset.graph['node_feat'])), dataset.label.unsqueeze(1)
        # return adj_low, features, labels
    elif dataset_name == 'Penn94':
        dataset = load_fb100_dataset('Penn94')
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label
    elif dataset_name == 'arxiv-year':
        dataset = load_arxiv_year_dataset()
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label
    elif dataset_name == 'pokec':
        dataset = load_pokec_mat()
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label
    elif dataset_name == 'snap-patents':
        dataset = load_snap_patents_mat()
        print('Done Loading...')
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        print('Done To Undirected...')
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label
    elif dataset_name == "genius":
        dataset = load_genius()
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label
    elif dataset_name == "twitch-gamer":
        dataset = load_twitch_gamer_dataset() 
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label
    elif dataset_name == "wiki":
        dataset = load_wiki()
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                'out1_node_feature_label.txt')

        G = nx.DiGraph().to_undirected()
        graph_node_features_dict = {}
        graph_labels_dict = {}
        
        
        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])
                    
                    
                    
        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    
    # features = preprocess_features(features)
    features = th.FloatTensor(features).to(device)
    labels = th.LongTensor(labels).to(device)
    
    if sage_data==True:
        if dataset_name in {'yelp-chi', 'deezer-europe'}:
            g = dgl.DGLGraph(adj+sp.eye(N))#.to(device)
        else:
            g = dgl.DGLGraph(adj+sp.eye(adj.shape[0]))#.to(device)
        # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
        g.ndata['features'] = features
        g.ndata['labels'] = labels
        degs = g.in_degrees().float()
        norm = th.pow(degs, -1).to(device)
        norm[th.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        return g, features, labels
    
    # g = adj
  
    # with np.load(splits_file_path) as splits_file:
    #     train_mask = splits_file['train_mask']
    #     val_mask = splits_file['val_mask']
    #     test_mask = splits_file['test_mask']
    
    # num_features = features.shape[1]
    # num_labels = len(np.unique(labels))
    # assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    
    # train_mask = th.BoolTensor(train_mask)
    # val_mask = th.BoolTensor(val_mask)
    # test_mask = th.BoolTensor(test_mask)

    # adj = normalize(adj+sp.eye(labels.shape[0])) #adj+sp.eye(labels.shape[0])#
    # g_high = sp.eye(g.shape[0]) - g
    if dataset_name in {'Crocodile-5', 'Crocodile-6'}:
        adj = torch.tensor(adj).to(torch.float32).to_sparse()
    else:
        print('From Matrix to Tensor...')
        adj = sparse_mx_to_torch_sparse_tensor(adj)#.to(device)
        print('Done Marrix to Tensor...')
    # g_high = sparse_mx_to_torch_sparse_tensor(g_high)
    print('Done Proccessing...')
    
    return adj, features, labels #g_high, f , train_mask, val_mask, test_mask

def data_split(idx, dataset_name):
    splits_file_path = 'splits/'+dataset_name+'_split_0.6_0.2_'+str(idx)+'.npz'
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)
    return train_mask, val_mask, test_mask

def normalize(mx, eqvar = None):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if eqvar:
        r_inv = np.power(rowsum, -1/eqvar).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
        
    else:
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    
def normalize_tensor(mx, symmetric = 0):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx,1)
    if symmetric == 0:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx
        
    else:
        r_inv = torch.pow(rowsum, -0.5).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(torch.mm(r_mat_inv, mx),r_mat_inv)
        return mx
    
def row_normalized_adjacency(adj):
    # adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm='l1', axis=1)
    # row_sum = np.array(adj.sum(1))
    # row_sum = (row_sum == 0)*1+row_sum
    # adj_normalized = adj/row_sum
    return sp.coo_matrix(adj_normalized)

def get_adj_high(adj_low):
    adj_high = -adj_low + sp.eye(adj_low.shape[0])
    return adj_high

def accuracy(labels, output):
    # print(output, output.dim)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    # if D==None:
    correct = correct.sum()
    return correct / len(labels)
    # else:
    #     D_correct = torch.mean(D[(correct==1)])
    #     D_incorrect = torch.mean(D[(correct==0)])
    #     correct = correct.sum()
    #     return correct / len(labels), [D_correct, D_incorrect]
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def dataset_edge_balance(adj, labels):
    """Measure the edge balance of each dataset"""
    num_class = labels.max().item() + 1
    num_nodes = labels.size()[0]
    num_nodes_per_class = np.zeros(num_class)
    edge_balance_per_class = np.zeros([num_class,2])
    for i in range(num_class):
        idx = np.where(labels.numpy()==i)[0]
        num_nodes_per_class[i] = idx.shape[0]
        edge_balance_per_class[i,0] = np.sum(adj[idx,:][:,idx])
        edge_balance_per_class[i,1] = np.sum(adj[idx, :][:, np.delete(np.arange(num_nodes), idx) ]) 
    
    return num_nodes_per_class, edge_balance_per_class

def random_disassortative_splits(labels, num_classes, training_percentage = 0.6):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    labels, num_classes = labels.cpu(), num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    percls_trn = int(round(training_percentage*(labels.size()[0]/num_classes)))
    val_lb = int(round(0.2*labels.size()[0]))
    # train_index = torch.cat([i[:int(len(i)*0.6)] for i in indices], dim=0)
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    # val_index = torch.cat([i[int(len(i)*0.6):int(len(i)*0.8)] for i in indices], dim=0)
    # test_index = torch.cat([i[int(len(i)*0.8):] for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    

    train_mask = index_to_mask(train_index, size=labels.size()[0])
    
    # val_mask = index_to_mask(val_index, size=labels.size()[0])
    # test_mask = index_to_mask(test_index, size=labels.size()[0])
    
    val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])
    
    return train_mask.to(device), val_mask.to(device), test_mask.to(device)


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def rand_train_test_idx(label, train_prop=.6, valid_prop=.2, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx

def load_fixed_splits(dataset, sub_dataset):
    """ loads saved fixed splits for dataset
    """
    name = dataset
    if sub_dataset and sub_dataset != 'None':
        name += f'-{sub_dataset}'

    if not os.path.exists(f'./data/splits/{name}-splits.npy'):
        assert dataset in splits_drive_url.keys()
        gdd.download_file_from_google_drive(
            file_id=splits_drive_url[dataset],
            dest_path=f'./data/splits/{name}-splits.npy', showsize=True)

    splits_lst = np.load(f'./data/splits/{name}-splits.npy', allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        
        """
        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None


def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}deezer-europe.mat')
    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset

def load_yelpchi_dataset():
    if not path.exists(f'{DATAPATH}YelpChi.mat'):
            gdd.download_file_from_google_drive(
                file_id= dataset_drive_url['yelp-chi'], \
                dest_path=f'{DATAPATH}YelpChi.mat', showsize=True) 
    fulldata = scipy.io.loadmat(f'{DATAPATH}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]
    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset

def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset

def load_arxiv_year_dataset(nclass=5):
    filename = 'arxiv-year'
    dataset = NCDataset(filename)
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    label = even_quantile_labels(
        dataset.graph['node_year'].flatten(), nclass, verbose=False)
    dataset.label = torch.as_tensor(label).reshape(-1, 1)
    return dataset

def load_pokec_mat():
    """ requires pokec.mat
    """
    if not path.exists(f'{DATAPATH}pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['pokec'], \
            dest_path=f'{DATAPATH}pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_snap_patents_mat(nclass=5):
    if not path.exists(f'{DATAPATH}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['snap-patents'], \
            dest_path=f'{DATAPATH}snap_patents.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{DATAPATH}snap_patents.mat')

    dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset

def load_genius():
    filename = 'genius'
    dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'data/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['twitch-gamer_feat'],
            dest_path=f'{DATAPATH}twitch-gamer_feat.csv', showsize=True)
    if not path.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['twitch-gamer_edges'],
            dest_path=f'{DATAPATH}twitch-gamer_edges.csv', showsize=True)
    
    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_wiki():

    if not path.exists(f'{DATAPATH}wiki_features2M.pt'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['wiki_features'], \
            dest_path=f'{DATAPATH}wiki_features2M.pt', showsize=True)
    
    if not path.exists(f'{DATAPATH}wiki_edges2M.pt'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['wiki_edges'], \
            dest_path=f'{DATAPATH}wiki_edges2M.pt', showsize=True)

    if not path.exists(f'{DATAPATH}wiki_views2M.pt'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['wiki_views'], \
            dest_path=f'{DATAPATH}wiki_views2M.pt', showsize=True)


    dataset = NCDataset("wiki") 
    features = torch.load(f'{DATAPATH}wiki_features2M.pt')
    edges = torch.load(f'{DATAPATH}wiki_edges2M.pt').T
    row, col = edges
    print(f"edges shape: {edges.shape}")
    label = torch.load(f'{DATAPATH}wiki_views2M.pt') 
    num_nodes = label.shape[0]

    print(f"features shape: {features.shape[0]}")
    print(f"Label shape: {label.shape[0]}")
    dataset.graph = {"edge_index": edges, 
                     "edge_feat": None, 
                     "node_feat": features, 
                     "num_nodes": num_nodes}
    dataset.label = label 
    return dataset 


def read_SuperGAT_crocodile():
    # 6 classes label from SuperGAT
    # download wikipedia.zip https://snap.stanford.edu/data/wikipedia-article-networks.html
    # crocodile: if do not remove self-loops, and use directed edge: 180020 edges in total
    # remove self-loops and make it undirected: 341546 edges in total
    num_vocab = 13183
    edge_index = pandas.read_csv('./new_data/crocodile/musae_crocodile_edges.csv', sep=',', header=None, skiprows=1, dtype=np.int64)
    edge_index = torch.from_numpy(edge_index.values).t()
    edge_index = remove_self_loops(edge_index)[0] # remove self-loops
    edge_index = to_undirected(edge_index) # make edges undirected
    adj = to_dense_adj(edge_index)[0].numpy()
    y = pandas.read_csv('./new_data/crocodile/musae_crocodile_target.csv', sep=",", dtype=np.float32)
    y = y.sort_values(by=["id"], ascending=True)
    y = torch.from_numpy(y.values[:, 1])
    y = np.log10(y).numpy()
    with open('./new_data/crocodile/musae_crocodile_features.json', "r") as feature_json:
        x_json = json.load(feature_json)
        x = np.zeros(shape=(len(x_json), num_vocab))
        for k_str, v_list in x_json.items():
            x[int(k_str), np.asarray(v_list)] += 1
        x = torch.from_numpy(x).float()
    labels = np.digitize(y, [2, 2.5, 3, 3.5, 4]) # class = 6
    features = x
    print("Load SuperGAT crocodile | %d nodes | %d edges | %d features | %d classes"%(adj.shape[0], adj.sum(), len(features[0]), max(labels)+1))
    return adj, features, labels


def read_WGCN_crocodile():
    # 5 classes labels from WGCN
    graph_adjacency_list_file_path = os.path.join('new_data/crocodile/out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('new_data/crocodile/out1_node_feature_label.txt')
    G = nx.DiGraph().to_undirected()
    graph_node_features_dict = {}
    graph_labels_dict = {}
    with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
        header = graph_node_features_and_labels_file.readline()
        feature_dim = int((header.rstrip().split('	'))[4])
        for line in graph_node_features_and_labels_file:
            line = line.rstrip().split('	')
            assert (len(line) == 3)
            assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
            features = np.zeros(feature_dim, dtype=np.uint8)
            if len(line[1]) > 0:
                values = np.array(line[1].split(','), dtype=np.uint8)
                for i in range(len(values)):
                    features[values[i]] = 1
            graph_node_features_dict[int(line[0])] = features
            graph_labels_dict[int(line[0])] = int(line[2])
    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                            label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                            label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))
    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = np.array(adj.todense())
    adj = adj - np.diag(np.diag(adj)) # remove self-loop
    features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    print("Load WGCN crocodile | %d nodes | %d edges | %d features | %d classes"%(adj.shape[0], adj.sum(), len(features[0]), max(labels)+1))
    return adj, features, labels

# after remove self-loop and make edges undirected, the adj for crocodile from torch_geometric and snap.stanford/SuperGAT/WGCN are identical
def load_geometric_crocodile():
    "output row-normalized feature; remove self-loop; y is continuous"
    # remove self-loops: 341546 edges in total; already undirected
    from torch_sparse import coalesce
    data = np.load('./new_data/crocodile/crocodile.npz', 'r', allow_pickle=True)
    x = torch.from_numpy(data['features']).to(torch.float)
    edge_index = torch.from_numpy(data['edges']).to(torch.long)
    edge_index = edge_index.t().contiguous()
    edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
    edge_index = remove_self_loops(edge_index)[0]
    adj = to_dense_adj(edge_index)[0].numpy()
    y = torch.from_numpy(data['target']).to(torch.float)
    print("Load geometric crocodile | %d nodes | %d edges | %d features | continuous y"%(adj.shape[0], adj.sum(), len(x[0])))
    features = x
    return adj, features, y

def WGCN_split(labels, train_percentage = 0.6, val_percentage = 0.2):   
    train_and_val_index, test_index = next(
        ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
            np.empty_like(labels), labels))
    train_index, val_index = next(
        ShuffleSplit(n_splits=1, train_size=train_percentage / (train_percentage + val_percentage)).split(
            np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
    train_index = train_and_val_index[train_index]
    val_index = train_and_val_index[val_index]
    return train_index, val_index, test_index

# SuperGAT split: The split is 20-per-class/30-per-class/rest from "Pitfalls of graph neural network evaluation arXiv:1811.05868"
# train, valid, test: 120, 180, 11331 for crocodile
# see: https://github.com/dongkwan-kim/SuperGAT/blob/master/SuperGAT/data_utils.py#L9
def SuperGAT_split(num_nodes, num_classes, labels, num_train_per_class=20, num_val_per_class=30, seed=12345):
    train_mask = torch.zeros([num_nodes], dtype=torch.bool)
    val_mask = torch.zeros([num_nodes], dtype=torch.bool)
    test_mask = torch.ones([num_nodes], dtype=torch.bool)
    random.seed(seed)
    for c in range(num_classes):
        samples_idx = (labels == c).nonzero().squeeze()
        perm = list(range(samples_idx.size(0)))
        random.shuffle(perm)
        perm = torch.as_tensor(perm).long()
        train_mask[samples_idx[perm][:num_train_per_class]] = True
        val_mask[samples_idx[perm][num_train_per_class:num_train_per_class + num_val_per_class]] = True
    test_mask[train_mask] = False
    test_mask[val_mask] = False
    node_list = np.array(list(range(num_nodes)))
    train_index = node_list[np.array(train_mask)]
    val_index = node_list[np.array(val_mask)]
    test_index = node_list[np.array(test_mask)]
    return train_index, val_index, test_index

def load_fb100(filename):
    # e.g. filename = Rutgers89 or Cornell5 or Wisconsin87 or Amherst41
    # columns are: student/faculty, gender, major,
    #              second major/minor, dorm/house, year/ high school
    # 0 denotes missing entry
    mat = scipy.io.loadmat(DATAPATH + 'facebook100/' + filename + '.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata

def load_twitch(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    filepath = f"data/twitch/{lang}"
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(f"{filepath}/musae_{lang}_target.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2]=="True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=np.int)
    with open(f"{filepath}/musae_{lang}_edges.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f"{filepath}/musae_{lang}_features.json", 'r') as f:
        j = json.load(f)
    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)
    inv_node_ids = {node_id:idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]
    
    n = label.shape[0]
    A = scipy.sparse.csr_matrix((np.ones(len(src)), 
                                 (np.array(src), np.array(targ))),
                                shape=(n,n))
    features = np.zeros((n,3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = features[:, np.sum(features, axis=0) != 0] # remove zero cols
    new_label = label[reorder_node_ids]
    label = new_label
    
    return A, label, features


def load_pokec():
    pathname = f"{DATAPATH}pokec/"
    node_filename = pathname + 'soc-pokec-profiles.txt'
    with open(node_filename, 'r') as f:
        user_lst = f.readlines()
    label = []
    for user in user_lst:
        gender = user.split('\t')[3]
        gender = int(gender) if gender != 'null' else -1
        label.append(gender)
    label = np.array(label)
    edge_filename = pathname + 'soc-pokec-relationships.txt'
    src = []
    targ = []
    with open(edge_filename, 'r') as f:
        count = 0
        for row in f:
            elts = row.split()
            src.append(int(elts[0]))
            targ.append(int(elts[1]))
            count += 1
            if count % 3000000 == 0:
                print("Loading edges:", count)
    src = np.array(src) - 1
    targ = np.array(targ) - 1
    A = scipy.sparse.csr_matrix((np.ones(len(src)), (src, targ)))
    return A, label

def load_twitch_gamer(nodes, task="dead_account"):
    nodes = nodes.drop('numeric_id', axis=1)
    nodes['created_at'] = nodes.created_at.replace('-', '', regex=True).astype(int)
    nodes['updated_at'] = nodes.updated_at.replace('-', '', regex=True).astype(int)
    one_hot = {k: v for v, k in enumerate(nodes['language'].unique())}
    lang_encoding = [one_hot[lang] for lang in nodes['language']]
    nodes['language'] = lang_encoding
    
    if task is not None:
        label = nodes[task].to_numpy()
        features = nodes.drop(task, axis=1).to_numpy()
    
    return label, features

def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def gen_normalized_adjs(dataset):
    """ returns the normalized adjacency matrix
    """
    dataset.graph['edge_index'] = add_self_loops(dataset.graph['edge_index'])[0]
    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0
    DAD = D_isqrt.view(-1,1) * adj * D_isqrt.view(1,-1)
    DA = D_isqrt.view(-1,1) * D_isqrt.view(-1,1) * adj
    AD = adj * D_isqrt.view(1,-1) * D_isqrt.view(1,-1)
    return DAD, DA, AD


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                                
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

@torch.no_grad()
def evaluate(output, labels, split_idx, eval_func):
    acc = eval_func(labels[split_idx], output[split_idx])
    return acc

def Prop(output, labels, idx_train, idx_val, idx_test, sample_max = 5000):
    # adj = (adj_low>0).float()
    # degree = torch.sum(adj,1)
    # centers = torch.zeros(labels.max()+1, output.shape[1])
    # std = torch.zeros(labels.max()+1, output.shape[1])
    # var = torch.zeros(labels.max()+1, output.shape[1])
    # ave_degree =  torch.zeros(labels.max()+1, 1)
    # num_node_class = torch.zeros(labels.max()+1,1)
    labels_train, labels_val, labels_test = labels[idx_train], labels[idx_val], labels[idx_test]
    output_train, output_val, output_test = output[idx_train], output[idx_val], output[idx_test]
    # output_val_train, labels_val_train = torch.cat([output_val,output_train], dim=1), torch.hstack([labels_val,labels_train])
    output_val_train = torch.cat([output_val,output_train], dim=0)
    # print(output_val.shape,output_train.shape)
    # if labels_val.dim() == 1:
    #     labels_val_train = torch.hstack([labels_val,labels_train])
    if labels_val.dim() == 2:
        # print(labels_val.shape,labels_train.shape)
        # print(labels_val,labels_train)
        #torch.cat([labels_val,labels_train], dim=0)
        labels_train, labels_val, labels_test = labels_train.flatten(), labels_val.flatten(), labels_test.flatten()
    labels_val_train = torch.cat([labels_val,labels_train]) ##torch.hstack([labels_val,labels_train])
    
    #test-train similarity
    # num_inner = torch.sum((labels_test.reshape(labels_test.shape[0],-1) == labels_train.reshape(-1, labels_train.shape[0])),1)
    # num_inter = torch.sum((labels_test.reshape(labels_test.shape[0],-1) != labels_train.reshape(-1, labels_train.shape[0])),1)
    
    # output_inner = torch.sum(torch.cdist(output_test,output_train,2)*(labels_test.reshape(labels_test.shape[0],-1) == labels_train.reshape(-1, labels_train.shape[0])),1)/num_inner
    # output_inter = torch.sum(torch.cdist(output_test,output_train,2)*(labels_test.reshape(labels_test.shape[0],-1) != labels_train.reshape(-1, labels_train.shape[0])),1)/num_inter
    # output_inner_std = torch.std(torch.cdist(output_test,output_train,2)*(labels_test.reshape(labels_test.shape[0],-1) == labels_train.reshape(-1, labels_train.shape[0])),1)
    # output_inter_std = torch.std(torch.cdist(output_test,output_train,2)*(labels_test.reshape(labels_test.shape[0],-1) != labels_train.reshape(-1, labels_train.shape[0])),1)
    # nodewise-classwise similarity
    # similarity_matrix = torch.zeros(labels.shape[0], labels.max()+1)
    # for i in range(labels.max()+1):
    #     similarity_matrix[:,i] = torch.mean(torch.cdist(output,output,2)[:,labels==i])
    ##Node-wise t-test hypothesis testing
    n_good = 0
    nnodes_val  = labels_val.shape[0]
    if nnodes_val >= sample_max:
        print("number of validation nodes is over ", sample_max)
        val_sample = random.sample(list(np.arange(nnodes_val)), sample_max)
        output_val = output_val[val_sample]
        labels_val = labels_val[val_sample]
    # dist_val_valtrain = torch.cdist(output_val,output_val_train,2)
    for i in range(labels_val.shape[0]):
        # a = torch.cdist(output_val,output_val,2)[i,:][labels_val[i] == labels_val]
        # b = torch.cdist(output_val,output_val,2)[i,:][labels_val[i] != labels_val]
        
        a = torch.cdist(output_val,output_val_train,2)[i,:][labels_val[i] == labels_val_train]
        b = torch.cdist(output_val,output_val_train,2)[i,:][labels_val[i] != labels_val_train]
        
        stat, p = ttest_ind(a.detach(), b.detach(), axis=0, equal_var=False, nan_policy='propagate') #, alternative='less'
        n_good = n_good + (p<0.025)
    
    
    

    return n_good/nnodes_val #ind_output#/ind_features


   
dataset_drive_url = {
    'twitch-gamer_feat': '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges': '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',  # Wiki 1.9M
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',  # Wiki 1.9M
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'  # Wiki 1.9M
}

splits_drive_url = {
    'snap-patents': '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
    'pokec': '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}

import math
import random
import time

import numpy as np
import scipy
import torch
from scipy.stats import ttest_ind
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import GaussianNB

from utils.util_funcs import random_disassortative_splits

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
device = torch.device(device)

pi = math.pi


def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr


def edge_homophily(adj, label):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    """
    adj = (adj > 0).float()
    adj = adj - torch.diag(torch.diag(adj))
    label_adj = torch.mm(label, label.transpose(0, 1))
    edge_hom = torch.sum(label_adj * adj) / torch.sum(adj)

    return edge_hom


def generalized_edge_homophily(adj, features, label, sample_max=20000, iteration=100):
    adj = (adj > 0).float()
    nnodes = label.shape[0]
    if nnodes < sample_max:
        sim = torch.tensor(cosine_similarity(features.cpu(), features.cpu())).to(device)
        sim[torch.isnan(sim)] = 0
        adj = adj - torch.diag(torch.diag(adj))
        g_edge_homo = torch.sum(sim * adj) / torch.sum(adj)

        return g_edge_homo
    else:
        edge_homo = np.zeros(iteration)
        for i in range(iteration):
            val_sample = torch.tensor(random.sample(list(np.arange(nnodes)), int(sample_max)))
            sim = torch.tensor(cosine_similarity(features[val_sample].cpu(), features[val_sample].cpu())).to(device)
            sim[torch.isnan(sim)] = 0

            adj = adj.to_dense()[val_sample, :][:, val_sample]
            adj = adj - torch.diag(torch.diag(adj))

            edge_homo[i] = torch.sum(sim * adj) / torch.sum(adj)

        return np.mean(edge_homo)


def node_homophily(A, labels):
    """ average of homophily for each node
    """

    src_node, targ_node = A.nonzero()[:, 0], A.nonzero()[:, 1]
    edge_idx = torch.tensor(np.vstack((src_node.cpu(), targ_node.cpu())), dtype=torch.long).contiguous().to(device)
    labels = labels.clone().detach()
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)


def node_homophily_edge_idx(edge_index, labels, num_nodes):
    """ edge_idx is 2 x(number edges) """

    hs = torch.zeros(num_nodes).to(device)
    degs = torch.bincount(edge_index[0, :]).float().to(device)
    matches = (labels[edge_index[0, :]] == labels[edge_index[1, :]]).float().to(device)
    hs = hs.scatter_add(0, edge_index[0, :], matches) / degs
    return hs[degs != 0].mean()


def compact_matrix_edge_idx(edge_index, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """

    src_node, targ_node = edge_index[:, 0], edge_index[:, 1]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max() + 1
    H = torch.zeros((c, c)).to(device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        H[k, :].scatter_add_(src=torch.ones_like(add_idx).to(H.dtype), dim=-1, index=add_idx)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H


def our_measure(A, label):
    """ 
    our measure \hat{h}
    treats negative labels as unlabeled 
    """
    A = A - torch.diag(torch.diag(A))
    A = A + torch.diag((torch.sum(A, 1) == 0).float())
    edge_index = A.nonzero()
    label = label.squeeze()
    c = label.max() + 1
    H = compact_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k, k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c - 1
    return val


def class_distribution(A, labels):
    edge_index = A.to_sparse().coalesce().indices()
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    deg = src_node.unique(return_counts=True)[1]
    deg = deg.to(device)

    # remove self-loop
    deg = deg - 1
    edge_index = remove_self_loops(A.to_sparse().coalesce().indices())[0]
    src_node, targ_node = edge_index[0, :], edge_index[1, :]

    labels = labels.squeeze()
    p = labels.unique(return_counts=True)[1] / labels.shape[0]
    p_bar = torch.zeros(labels.max() + 1, device=labels.device)
    pc = torch.zeros((labels.max() + 1, labels.max() + 1), device=labels.device)
    for i in range(labels.max() + 1):
        p_bar[i] = torch.sum(deg[torch.where(labels == i)])

        for j in range(labels.max() + 1):
            pc[i, j] = torch.sum(labels[targ_node[torch.where(labels[src_node] == i)]] == j)
    p_bar, pc = p_bar / torch.sum(deg), pc / torch.sum(deg)
    p_bar[torch.where(p_bar == 0)], pc[torch.where(pc == 0)] = 1e-8, 1e-8
    return p, p_bar, pc


def adjusted_homo(A, label):
    p, p_bar, pc = class_distribution(A, torch.argmax(label, 1))
    edge_homo = edge_homophily(A, label)
    adj_homo = (edge_homo - torch.sum(p_bar ** 2)) / (1 - torch.sum(p_bar ** 2))

    return adj_homo


def label_informativeness(A, label):
    p, p_bar, pc = class_distribution(A, torch.argmax(label, 1))
    LI = 2 - torch.sum(pc * torch.log(pc)) / torch.sum(p_bar * torch.log(p_bar))
    return LI


def similarity(features, adj, label, NTK=None, hard=None, LP=1, ifsum=1, idx_train=None):
    if NTK:
        inner_prod = torch.clamp(torch.mm(features, features.transpose(0, 1)), 0, 1)

        inner_prod = (inner_prod * (torch.pi - torch.acos(inner_prod))) / (2 * torch.pi)
        inner_prod = torch.mm(torch.mm(adj, inner_prod), adj.transpose(0, 1))
    else:
        inner_prod = torch.mm(torch.mm(adj, features), torch.mm(adj, features).transpose(0, 1))

    if str(type(idx_train)) == '<class \'NoneType\'>':
        labels = torch.argmax(label, 1)
        weight_matrix = (torch.zeros(adj.clone().detach().size(0), labels.clone().detach().max() + 1))
    else:
        labels = torch.argmax(label, 1)[idx_train]
        label = label[idx_train, :]
        weight_matrix = (torch.zeros(idx_train.shape[0], labels.clone().detach().max() + 1))
        inner_prod = inner_prod[idx_train, :][:, idx_train]
    for i in range(labels.max() + 1):
        # Think about using torch.sum or torch.mean
        if ifsum == 1:
            weight_matrix[:, i] = torch.sum(inner_prod[:, labels == i], 1)
        else:
            weight_matrix[:, i] = torch.mean(inner_prod[:, labels == i], 1)
    if hard == None:

        if ifsum == 1:
            nnodes = labels.shape[0]
            degs_label = torch.sum(torch.mm(label, label.transpose(0, 1)), 1)
        else:
            nnodes = labels.max() + 1
            degs_label = 1
        if LP == 1:

            LAF_ratio = (weight_matrix[np.arange(labels.size(0)), labels] / degs_label) / (
                    (torch.sum(weight_matrix, 1) - weight_matrix[np.arange(labels.size(0)), labels]) / (
                    nnodes - degs_label))
            LAF_ratio[torch.isnan(LAF_ratio)] = 0
            return torch.mean((LAF_ratio >= 1).float())
        else:
            return torch.mean(((torch.sum(weight_matrix - weight_matrix * label, 1) <= 0) & (
                    torch.sum(weight_matrix * label, 1) >= 0)).float())
    else:
        if LP == 1:
            return torch.mean(torch.argmax(weight_matrix, 1).eq(labels).float())
        else:
            return torch.mean(((torch.max(weight_matrix - weight_matrix * label, 1)[0] <= 0.) & (
                    torch.sum(weight_matrix * label, 1) >= 0)).float())


def gntk_homophily_(features, adj, sample, n_layers):
    if features.device != sample.device:
        features = features.to(sample.device)
    if adj.device != sample.device:
        adj = adj.to(sample.device)
        
    eps = 1e-8
    nnodes = features.shape[0]
    G_gram = torch.mm(torch.spmm(adj, features)[sample, :], torch.transpose(torch.spmm(adj, features)[sample, :], 0, 1))
    G_norm = torch.sqrt(torch.diag(G_gram)).reshape(-1, 1) * torch.sqrt(torch.diag(G_gram)).reshape(1, -1)
    G_norm = (G_norm > eps) * G_norm + eps * (G_norm <= eps)
    if n_layers == 1:
        arccos = torch.acos(torch.div(G_gram, G_norm))
        sqrt = torch.sqrt(torch.square(G_norm) - torch.square(G_gram))
        arccos[torch.isnan(arccos)], sqrt[torch.isnan(sqrt)] = 0, 0
        K_G = 1 / pi * (G_gram * (pi - arccos) + sqrt)
    else:
        K_G = G_gram

    gram = torch.mm(features[sample, :], torch.transpose(features[sample, :], 0, 1))
    norm = torch.sqrt(torch.diag(gram)).reshape(-1, 1) * torch.sqrt(torch.diag(gram)).reshape(1, -1)
    norm = (norm > eps) * norm + eps * (norm <= eps)
    if n_layers == 1:
        arccos = torch.acos(torch.div(gram, norm))
        sqrt = torch.sqrt(torch.square(norm) - torch.square(gram))
        arccos[torch.isnan(arccos)], sqrt[torch.isnan(sqrt)] = 0, 0
        K_X = 1 / pi * (gram * (pi - arccos) + sqrt)
    else:
        K_X = gram

    return K_G / 2, K_X / 2


def classifier_based_performance_metric(features, adj, labels, sample_max, rcond=1e-15, base_classifier='kernel_reg1',
                                        epochs=100):
    if features.device.type != labels.device.type:
        features = features.to(labels.device) # Ensure features is on the same device as labels
    
    if adj.device.type != labels.device.type:
        adj = adj.to(labels.device) # Ensure adj is on the same device as labels

    
    nnodes = (labels.shape[0])
    if labels.dim() > 1:
        labels = labels.flatten()
    G_results, X_results, diff_results, G_good_p_results, X_good_p_results = torch.zeros(epochs), torch.zeros(
        epochs), torch.zeros(epochs), torch.zeros(epochs), torch.zeros(epochs)
    t_time = time.time()
    for j in range(epochs):

        if nnodes <= sample_max:
            sample = np.arange(nnodes)
            label_onehot = torch.eye(labels.max() + 1)[labels].cpu()
            labels_sample = labels.cpu()
        else:
            sample, _, _ = random_disassortative_splits(labels, labels.max() + 1, sample_max / nnodes)
            label_onehot = torch.eye(labels.max() + 1, device=device)[labels][sample, :].cpu()            
            labels_sample = labels[sample]
        idx_train, idx_val, idx_test = random_disassortative_splits(labels_sample, labels_sample.max() + 1)
        idx_val = idx_val + idx_test

        # Kernel Regression based p-values
        if base_classifier in {'kernel_reg0', 'kernel_reg1'}:
            nlayers = 0 if base_classifier == 'kernel_reg0' else 1
            K_graph, K = gntk_homophily_(features, adj, sample, n_layers=nlayers)
            K_graph_train_train, K_train_train = K_graph[idx_train, :][:, idx_train], K[idx_train, :][:, idx_train]
            K_graph_val_train, K_val_train = K_graph[idx_val, :][:, idx_train], K[idx_val, :][:, idx_train]
            Kreg_G, Kreg_X = K_graph_val_train.cpu() @ (
                    torch.tensor(np.linalg.pinv(K_graph_train_train.cpu().numpy())) @ label_onehot[
                idx_train.to(label_onehot.device)]), K_val_train.cpu() @ (
                                     torch.tensor(np.linalg.pinv(K_train_train.cpu().numpy())) @ label_onehot.cpu()[
                                 idx_train.to(label_onehot.device)])

            labels_sample = labels_sample.to(Kreg_G.device) 
            idx_val = idx_val.to(Kreg_G.device)
            diff_results[j] = (torch.mean(Kreg_G.argmax(1).eq(labels_sample[idx_val]).float()) > torch.mean(
                Kreg_X.argmax(1).eq(labels_sample[idx_val]).float()))
            G_results[j] = torch.mean(Kreg_G.argmax(1).eq(labels_sample[idx_val]).float())
            X_results[j] = torch.mean(Kreg_X.argmax(1).eq(labels_sample[idx_val]).float())
        elif base_classifier == 'gnb':
            #  Gaussian Naive Bayes model
            X = features[sample].cpu()
            X_agg = torch.spmm(adj, features)[sample].cpu()

            X_gnb, G_gnb = GaussianNB(), GaussianNB()
            X_gnb.fit(X[idx_train], labels_sample[idx_train])
            G_gnb.fit(X_agg[idx_train], labels_sample[idx_train])

            X_pred = torch.tensor(X_gnb.predict(X[idx_val]))
            G_pred = torch.tensor(G_gnb.predict(X_agg[idx_val]))

            diff_results[j] = (torch.mean(G_pred.eq(labels_sample[idx_val]).float()) > torch.mean(
                X_pred.eq(labels_sample[idx_val]).float()))
            G_results[j] = torch.mean(G_pred.eq(labels_sample[idx_val]).float())
            X_results[j] = torch.mean(X_pred.eq(labels_sample[idx_val]).float())
        else:
            #  SVM based p-values
            X = features[sample].cpu()
            X_agg = torch.spmm(adj, features)[sample].cpu()
            if base_classifier == 'svm_rbf':
                G_svm = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_agg[idx_train], labels_sample[idx_train])
                X_svm = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X[idx_train], labels_sample[idx_train])
            elif base_classifier == 'svm_poly':
                G_svm = svm.SVC(kernel='poly', degree=3, C=1).fit(X_agg[idx_train], labels_sample[idx_train])
                X_svm = svm.SVC(kernel='poly', degree=3, C=1).fit(X[idx_train], labels_sample[idx_train])
            elif base_classifier == 'svm_linear':
                G_svm = svm.SVC(kernel='linear').fit(X_agg[idx_train], labels_sample[idx_train])
                X_svm = svm.SVC(kernel='linear').fit(X[idx_train], labels_sample[idx_train])

            G_pred = torch.tensor(G_svm.predict(X_agg[idx_val]))
            X_pred = torch.tensor(X_svm.predict(X[idx_val]))
            diff_results[j] = (torch.mean(G_pred.eq(labels_sample[idx_val]).float()) > torch.mean(
                X_pred.eq(labels_sample[idx_val]).float()))
            G_results[j] = torch.mean(G_pred.eq(labels_sample[idx_val]).float())
            X_results[j] = torch.mean(X_pred.eq(labels_sample[idx_val]).float())

    if scipy.__version__ == '1.4.1':
        g_aware_good_stats, g_aware_good_p = ttest_ind(X_results.detach().cpu(), G_results.detach().cpu(), axis=0,
                                                       equal_var=False,
                                                       nan_policy='propagate')  # ttest_1samp(diff_results,0.5)
    else:
        g_aware_good_stats, g_aware_good_p = ttest_ind(X_results.detach().cpu(), G_results.detach().cpu(), axis=0,
                                                       equal_var=False, nan_policy='propagate')

    if torch.mean(diff_results) <= 0.5:
        g_aware_good_p = g_aware_good_p / 2

    else:
        g_aware_good_p = 1 - g_aware_good_p / 2

    return g_aware_good_p

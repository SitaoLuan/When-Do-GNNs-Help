import os
import pickle as pkl
import sys
from os import path

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from google_drive_downloader import GoogleDriveDownloader as gdd
from sklearn.preprocessing import normalize as sk_normalize

from utils.datasets import load_fb100_dataset, load_wiki, load_genius, load_deezer_dataset, load_yelpchi_dataset, \
    load_arxiv_year_dataset, load_twitch_gamer_dataset, load_pokec_mat, load_snap_patents_mat, read_WGCN_crocodile, \
    read_SuperGAT_crocodile

if torch.cuda.is_available():
    from torch_geometric.utils import to_dense_adj, contains_self_loops, remove_self_loops, \
        to_dense_adj, to_undirected

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

sys.setrecursionlimit(99999)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = (1 / rowsum).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = (1 / rowsum).flatten()
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
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
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
        data = eval(dataset + "(root = '" + cur.replace("\\",
                                                        "/") + "/torch_geometric_data/" + dataset + "'," + "name = '" + name + "')")
    # e.g. Coauthor(root='...', name = 'CS')

    edge = data[0].edge_index
    if contains_self_loops(edge):
        edge = remove_self_loops(edge)[0]
        print("Original data contains self-loop, it is now removed")

    adj = to_dense_adj(edge)[0].numpy()

    print("Nodes: %d, edges: %d, features: %d, classes: %d. \n" % (
        len(adj[0]), len(edge[0]) / 2, len(data[0].x[0]), len(np.unique(data[0].y))))

    mask = np.transpose(adj) != adj
    col_sum = adj.sum(axis=0)
    print("Check adjacency matrix is sysmetric: %r" % (mask.sum().item() == 0))
    print("Check the number of isolated nodes: %d" % ((col_sum == 0).sum().item()))
    print("Node degree Max: %d, Mean: %.4f, SD: %.4f" % (col_sum.max(), col_sum.mean(), col_sum.std()))

    return adj, data[0].x.numpy(), data[0].y.numpy()


def full_load_data(dataset_name):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels = load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
    elif dataset_name in {'CitationFull_dblp', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Computers', 'Amazon_Photo'}:
        dataset, name = dataset_name.split("_")
        adj, features, labels = load_torch_geometric_data(dataset, name)

    elif dataset_name in {'Flickr', 'WikiCS'}:
        adj, features, labels = load_torch_geometric_data(dataset_name, None)

    else:
        graph_adjacency_list_file_path = os.path.join(path.dirname(path.abspath(__file__)), '../new_data', dataset_name,
                                                      'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join(path.dirname(path.abspath(__file__)), '../new_data',
                                                                dataset_name,
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
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)

    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)

    g = adj
    g = normalize(g + sp.eye(g.shape[0]))
    g_high = sp.eye(g.shape[0]) - g
    g = sparse_mx_to_torch_sparse_tensor(g)
    g_high = sparse_mx_to_torch_sparse_tensor(g_high)

    return g, g_high, features, labels


def full_load_data_large(dataset_name, sage_data=False):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels = load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
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
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label
    elif dataset_name == 'yelp-chi':
        dataset = load_yelpchi_dataset()
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label.unsqueeze(1)
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
        graph_adjacency_list_file_path = os.path.join(path.dirname(path.abspath(__file__)), '../new_data', dataset_name,
                                                      'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join(path.dirname(path.abspath(__file__)), '../new_data',
                                                                dataset_name,
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
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    features = th.FloatTensor(features).to(device)
    labels = th.LongTensor(labels).to(device)

    if sage_data == True:
        if dataset_name in {'yelp-chi', 'deezer-europe'}:
            g = dgl.DGLGraph(adj + sp.eye(N)).to(device)
        else:
            g = dgl.DGLGraph(adj + sp.eye(adj.shape[0])).to(device)
        # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
        g.ndata['features'] = features
        g.ndata['labels'] = labels
        degs = g.in_degrees().float()
        norm = th.pow(degs, -1).to(device)
        norm[th.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        return g, features, labels

    if dataset_name in {'Crocodile-5', 'Crocodile-6'}:
        adj = torch.tensor(adj).to(torch.float32).to_sparse()
    else:
        adj = sparse_mx_to_torch_sparse_tensor(adj)  # .to(device)
    print('Done Processing...')

    return adj, features, labels


def normalize_tensor(mx, symmetric=0):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, 1)
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
        mx = torch.mm(torch.mm(r_mat_inv, mx), r_mat_inv)
        return mx


def row_normalized_adjacency(adj):
    # adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm='l1', axis=1)
    # row_sum = np.array(adj.sum(1))
    # row_sum = (row_sum == 0)*1+row_sum
    # adj_normalized = adj/row_sum
    return sp.coo_matrix(adj_normalized)


def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


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
    row_sum = (row_sum == 0) * 1 + row_sum
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
    edge_balance_per_class = np.zeros([num_class, 2])
    for i in range(num_class):
        idx = np.where(labels.numpy() == i)[0]
        num_nodes_per_class[i] = idx.shape[0]
        edge_balance_per_class[i, 0] = np.sum(adj[idx, :][:, idx])
        edge_balance_per_class[i, 1] = np.sum(adj[idx, :][:, np.delete(np.arange(num_nodes), idx)])

    return num_nodes_per_class, edge_balance_per_class


def random_disassortative_splits(labels, num_classes, training_percentage=0.6):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    labels, num_classes = labels.cpu(), num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    percls_trn = int(round(training_percentage * (labels.size()[0] / num_classes)))
    val_lb = int(round(0.2 * labels.size()[0]))

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=labels.size()[0])
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


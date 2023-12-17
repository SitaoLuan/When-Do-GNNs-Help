import json
import os
from os import path

import networkx as nx
import numpy as np
import pandas
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from ogb.nodeproppred import NodePropPredDataset

if torch.cuda.is_available():
    import scipy.io
    from sklearn.preprocessing import label_binarize

    from torch_geometric.utils import to_dense_adj, remove_self_loops, \
        to_dense_adj, to_undirected

DATA_PATH = path.dirname(path.abspath(__file__)) + '/../data/'
DATASET_DRIVE_URL = {
    'twitch-gamer_feat': '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
    'twitch-gamer_edges': '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
    'wiki_views': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP',  # Wiki 1.9M
    'wiki_edges': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u',  # Wiki 1.9M
    'wiki_features': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'  # Wiki 1.9M
}


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
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
    deezer = scipy.io.loadmat(f'{DATA_PATH}deezer-europe.mat')
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
    if not path.exists(f'{DATA_PATH}YelpChi.mat'):
        gdd.download_file_from_google_drive(
            file_id=DATASET_DRIVE_URL['yelp-chi'], \
            dest_path=f'{DATA_PATH}YelpChi.mat', showsize=True)
    fulldata = scipy.io.loadmat(f'{DATA_PATH}YelpChi.mat')
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
    if not path.exists(f'{DATA_PATH}pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id=DATASET_DRIVE_URL['pokec'], \
            dest_path=f'{DATA_PATH}pokec.mat', showsize=True)
    print(f'{DATA_PATH}pokec.mat')
    fulldata = scipy.io.loadmat(f'{DATA_PATH}pokec.mat')

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
    if not path.exists(f'{DATA_PATH}snap_patents.mat'):
        p = DATASET_DRIVE_URL['snap-patents']
        print(f"Snap patents url: {p}")
        gdd.download_file_from_google_drive(
            file_id=DATASET_DRIVE_URL['snap-patents'], \
            dest_path=f'{DATA_PATH}snap_patents.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{DATA_PATH}snap_patents.mat')

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
    fulldata = scipy.io.loadmat(f'{DATA_PATH}genius.mat')

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
    if not path.exists(f'{DATA_PATH}twitch-gamer_feat.csv'):
        gdd.download_file_from_google_drive(
            file_id=DATASET_DRIVE_URL['twitch-gamer_feat'],
            dest_path=f'{DATA_PATH}twitch-gamer_feat.csv', showsize=True)
    if not path.exists(f'{DATA_PATH}twitch-gamer_edges.csv'):
        gdd.download_file_from_google_drive(
            file_id=DATASET_DRIVE_URL['twitch-gamer_edges'],
            dest_path=f'{DATA_PATH}twitch-gamer_edges.csv', showsize=True)

    edges = pd.read_csv(f'{DATA_PATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATA_PATH}twitch-gamer_feat.csv')
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
    if not path.exists(f'{DATA_PATH}wiki_features2M.pt'):
        gdd.download_file_from_google_drive(
            file_id=DATASET_DRIVE_URL['wiki_features'], \
            dest_path=f'{DATA_PATH}wiki_features2M.pt', showsize=True)

    if not path.exists(f'{DATA_PATH}wiki_edges2M.pt'):
        gdd.download_file_from_google_drive(
            file_id=DATASET_DRIVE_URL['wiki_edges'], \
            dest_path=f'{DATA_PATH}wiki_edges2M.pt', showsize=True)

    if not path.exists(f'{DATA_PATH}wiki_views2M.pt'):
        gdd.download_file_from_google_drive(
            file_id=DATASET_DRIVE_URL['wiki_views'], \
            dest_path=f'{DATA_PATH}wiki_views2M.pt', showsize=True)

    dataset = NCDataset("wiki")
    features = torch.load(f'{DATA_PATH}wiki_features2M.pt')
    edges = torch.load(f'{DATA_PATH}wiki_edges2M.pt').T
    row, col = edges
    print(f"edges shape: {edges.shape}")
    label = torch.load(f'{DATA_PATH}wiki_views2M.pt')
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
    edge_index = pandas.read_csv('./new_data/crocodile/musae_crocodile_edges.csv', sep=',', header=None, skiprows=1,
                                 dtype=np.int64)
    edge_index = torch.from_numpy(edge_index.values).t()
    edge_index = remove_self_loops(edge_index)[0]  # remove self-loops
    edge_index = to_undirected(edge_index)  # make edges undirected
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
    labels = np.digitize(y, [2, 2.5, 3, 3.5, 4])  # class = 6
    features = x
    print("Load SuperGAT crocodile | %d nodes | %d edges | %d features | %d classes" % (
        adj.shape[0], adj.sum(), len(features[0]), max(labels) + 1))
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
    adj = adj - np.diag(np.diag(adj))  # remove self-loop
    features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    print("Load WGCN crocodile | %d nodes | %d edges | %d features | %d classes" % (
        adj.shape[0], adj.sum(), len(features[0]), max(labels) + 1))
    return adj, features, labels


def load_fb100(filename):
    # e.g. filename = Rutgers89 or Cornell5 or Wisconsin87 or Amherst41
    # columns are: student/faculty, gender, major,
    #              second major/minor, dorm/house, year/ high school
    # 0 denotes missing entry
    mat = scipy.io.loadmat(DATA_PATH + 'facebook100/' + filename + '.mat')
    A = mat['A']
    metadata = mat['local_info']
    return A, metadata


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
    returns a np array of int class labels
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

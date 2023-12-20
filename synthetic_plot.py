# Plot the results of classifier-based performance metrics and baseline metrics on synthetic graphs

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from gnns_on_syn import gcn_mean, gcn_std
from gnns_on_syn import mlp1_mean, mlp1_std
from gnns_on_syn import mlp2_mean, mlp2_std
from gnns_on_syn import sgc_mean, sgc_std
from utils.homophily_plot import classifier_based_performance_metric, edge_homophily, node_homophily, our_measure, similarity, \
    adjusted_homo, label_informativeness, generalized_edge_homophily
from utils.util_funcs import normalize, full_load_data, preprocess_features

adj_low, adj_high, features, labels = full_load_data('citeseer')

graph_hvalue_set = [0.05, 0.1, 0.15, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2, 0.21, 0.22, 0.23, 0.24,
                    0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9][::-1]
dataset_name = ['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'film']
num_edge_same = 4000
epochs = 100

KR_L_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
KR_L_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))
KR_NL_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
KR_NL_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))

node_homo_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
node_homo_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))
edge_homo_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
edge_homo_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))
class_homo_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
class_homo_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))
soft_las_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
soft_las_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))
ge_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
ge_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))

adj_homo_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
adj_homo_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))
LI_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
LI_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))

nnpg_X_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
nnpg_X_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))
gntk2_X_mean = np.zeros((len(dataset_name), len(graph_hvalue_set)))
gntk2_X_std = np.zeros((len(dataset_name), len(graph_hvalue_set)))

soft_lafs_name = ['soft_lafs_cora', 'soft_lafs_citeseer', 'soft_lafs_pubmed', 'soft_lafs_chameleon',
                  'soft_lafs_squirrel', 'soft_lafs_film']
hard_lafs_name = ['hard_lafs_cora', 'hard_lafs_citeseer', 'hard_lafs_pubmed', 'hard_lafs_chameleon',
                  'hard_lafs_squirrel', 'hard_lafs_film']
soft_lafd_name = ['soft_lafd_cora', 'soft_lafd_citeseer', 'soft_lafd_pubmed', 'soft_lafd_chameleon',
                  'soft_lafd_squirrel', 'soft_lafd_film']
hard_lafd_name = ['hard_lafd_cora', 'hard_lafd_citeseer', 'hard_lafd_pubmed', 'hard_lafd_chameleon',
                  'hard_lafd_squirrel', 'hard_lafd_film']
soft_lfs_name = ['soft_lfs_cora', 'soft_lfs_citeseer', 'soft_lfs_pubmed', 'soft_lfs_chameleon', 'soft_lfs_squirrel',
                 'soft_lfs_film']
hard_lfs_name = ['hard_lfs_cora', 'hard_lfs_citeseer', 'hard_lfs_pubmed', 'hard_lfs_chameleon', 'hard_lfs_squirrel',
                 'hard_lfs_film']

for k, base_dataset in zip(range(len(dataset_name)), dataset_name):
    for i, graph_h_value in zip(range(len(graph_hvalue_set)), graph_hvalue_set):
        sample_max = 300 if base_dataset in {'chameleon', 'film'} else 500

        edge_homo = np.zeros(10)
        node_homo = np.zeros(10)
        class_homo = np.zeros(10)
        soft_las = np.zeros(10)
        adj_homo = np.zeros(10)
        LI = np.zeros(10)
        ge = np.zeros(10)

        KR_L = np.zeros(10)
        KR_NL = np.zeros(10)
        for sample in range(10):
            file = 'data_synthesis'  # 'cora''cora'#
            Path(f"./data_synthesis/features").mkdir(parents=True, exist_ok=True)
            features = torch.tensor(preprocess_features(torch.load(("./data_synthesis/features/{}/{}_{}.pt".format(
                base_dataset, base_dataset, sample))).clone().detach().float())).clone().detach()

            Path(f"./data_synthesis/{num_edge_same}/{graph_h_value}").mkdir(parents=True, exist_ok=True)
            adj = torch.load((
                f"./data_synthesis/{num_edge_same}/{graph_h_value}/adj_{graph_h_value}_{sample}.pt")).to_dense().clone().detach().float()
            label = ((torch.load((
                f"./data_synthesis/{num_edge_same}/{graph_h_value}/label_{graph_h_value}_{sample}.pt")).to_dense().clone().detach().float())).clone().detach()  #
            degree = torch.load((
                f"./data_synthesis/{num_edge_same}/{graph_h_value}/degree_{graph_h_value}_{sample}.pt")).to_dense().clone().detach().float()
            nnodes = adj.shape[0]
            adj = torch.tensor(normalize(adj + torch.eye(nnodes)))

            KR_L[sample] = classifier_based_performance_metric(features, adj, torch.argmax(label, 1),
                                                               sample_max=sample_max,
                                                               base_classifier='kernel_reg0',
                                                               epochs=epochs)
            KR_NL[sample] = classifier_based_performance_metric(features, adj, torch.argmax(label, 1),
                                                                sample_max=sample_max,
                                                                base_classifier='kernel_reg1',
                                                                epochs=epochs)

            edge_homo[sample] = edge_homophily(adj, label)
            node_homo[sample] = node_homophily(adj, torch.argmax(label, 1))
            class_homo[sample] = our_measure(adj, torch.argmax(label, 1))
            soft_las[sample] = similarity(label, adj, label, NTK=None, hard=None, LP=1)
            adj_homo[sample] = adjusted_homo(adj, label)
            LI[sample] = label_informativeness(adj, label)
            ge[sample] = generalized_edge_homophily(adj, features, label)

        print("Computing CPM for %s with edge homo: %.2f, KR_L: %.4f, KR_NL: %.4f" % (
            base_dataset, graph_h_value, np.mean(KR_L), np.mean(KR_NL)))
        print(
            "Computing CPM for %s with edge homo: %.2f, node homo: %.4f, "
            "class homo: %.4f, agg homo: %.4f, GE homo: %.4f, adj homo: %.4f, LI: %.4f" % (
                base_dataset, graph_h_value, np.mean(node_homo), np.mean(class_homo), np.mean(soft_las), np.mean(ge),
                np.mean(adj_homo), np.mean(LI)))

        KR_L_mean[k, i] = np.mean(KR_L)
        KR_L_std[k, i] = np.std(KR_L)
        KR_NL_mean[k, i] = np.mean(KR_NL)
        KR_NL_std[k, i] = np.std(KR_NL)

        edge_homo_mean[k, i] = np.mean(edge_homo)
        edge_homo_std[k, i] = np.std(edge_homo)
        node_homo_mean[k, i] = np.mean(node_homo)
        node_homo_std[k, i] = np.std(node_homo)
        class_homo_mean[k, i] = np.mean(class_homo)
        class_homo_std[k, i] = np.std(class_homo)
        soft_las_mean[k, i] = np.mean(soft_las)
        soft_las_std[k, i] = np.std(soft_las)
        adj_homo_mean[k, i] = np.mean(adj_homo)
        adj_homo_std[k, i] = np.std(adj_homo)
        LI_mean[k, i] = np.mean(LI)
        LI_std[k, i] = np.std(LI)
        ge_mean[k, i] = np.mean(ge)
        ge_std[k, i] = np.std(ge)

    plt.figure()
    gcn_results, gcn_std_results = gcn_mean[k], gcn_std[k]
    sgc_results, sgc_std_results = sgc_mean[k], sgc_std[k]
    mlp1_results, mlp1_std_results = mlp1_mean[k], mlp1_std[k]
    mlp2_results, mlp2_std_results = mlp2_mean[k], mlp2_std[k]

    KR_L_results, KR_NL_results = KR_L_mean[k, :], KR_NL_mean[k, :]
    KR_L_std_results, KR_NL_std_results = KR_L_std[k, :], KR_NL_std[k, :]

    graph_hvalue_set = np.asarray(graph_hvalue_set)

    plt.plot(graph_hvalue_set, gcn_results / 100, color='black')
    plt.plot(graph_hvalue_set, mlp2_results / 100, '--', color='black')

    plt.plot(graph_hvalue_set, sgc_results / 100, color='red')
    plt.plot(graph_hvalue_set, mlp1_results / 100, '--', color='red')

    plt.plot(graph_hvalue_set, KR_L_results, color='green')
    plt.plot(graph_hvalue_set, KR_NL_results, color='blue')

    plt.plot(graph_hvalue_set, 0.5 * np.ones(graph_hvalue_set.shape), '--', color='green')
    plt.plot(graph_hvalue_set, 0.05 * np.ones(graph_hvalue_set.shape), '--', color='orange')
    legends = ['GCN Performance', 'MLP-2 Performance', 'SGC Performance', 'MLP-1 Performance', 'KR_L', 'KR_NL', 'NT0.5',
               'SST0.05']
    plt.legend(legends, bbox_to_anchor=(1, -0.02), loc='lower right')

    plt.xlabel('Edge Homophily')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    fig_name = f"./plots/{base_dataset}_NNGP_homo_performance_comparison.pdf"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    plt.close()

    # plot the results for baseline homophily metrics
    plt.figure()

    node_homo_results, node_homo_std_results = node_homo_mean[k, :], node_homo_std[k, :]
    edge_homo_results, edge_homo_std_results = edge_homo_mean[k, :], edge_homo_std[k, :]
    class_homo_results, class_homo_std_results = class_homo_mean[k, :], class_homo_std[k, :]
    agg_homo_results, agg_homo_std_results = soft_las_mean[k, :], soft_las_std[k, :]
    adj_homo_results, adj_homo_std_results = adj_homo_mean[k, :], adj_homo_std[k, :]
    LI_results, LI_std_results = LI_mean[k, :], LI_std[k, :]
    ge_results, ge_std_results = ge_mean[k, :], ge_std[k, :]

    graph_hvalue_set = np.asarray(graph_hvalue_set)

    plt.plot(graph_hvalue_set, gcn_results / 100, color='black')
    plt.plot(graph_hvalue_set, mlp2_results / 100, '--', color='black')
    plt.fill_between(graph_hvalue_set, np.clip(gcn_results - gcn_std_results, 0, 100) / 100,
                     np.clip(gcn_results + gcn_std_results, 0, 100) / 100, alpha=0.7, color='black')
    plt.fill_between(graph_hvalue_set, np.clip(mlp2_results - mlp2_std_results, 0, 100) / 100,
                     np.clip(mlp2_results + mlp2_std_results, 0, 100) / 100, alpha=0.7, color='black')

    plt.plot(graph_hvalue_set, sgc_results / 100, color='red')
    plt.plot(graph_hvalue_set, mlp1_results / 100, '--', color='red')
    plt.fill_between(graph_hvalue_set, np.clip(sgc_results - sgc_std_results, 0, 100) / 100,
                     np.clip(sgc_results + sgc_std_results, 0, 100) / 100, alpha=0.7, color='red')
    plt.fill_between(graph_hvalue_set, np.clip(mlp1_results - mlp1_std_results, 0, 100) / 100,
                     np.clip(mlp1_results + mlp1_std_results, 0, 100) / 100, alpha=0.7, color='red')

    plt.plot(graph_hvalue_set, node_homo_results, color='orange')
    plt.plot(graph_hvalue_set, class_homo_results, color='pink')
    plt.plot(graph_hvalue_set, agg_homo_results, color='purple')
    plt.plot(graph_hvalue_set, ge_results, color='yellow')
    plt.plot(graph_hvalue_set, adj_homo_results, color='blue')
    plt.plot(graph_hvalue_set, LI_results, color='grey')

    plt.fill_between(graph_hvalue_set, np.clip(node_homo_results - node_homo_std_results, 0, 1),
                     np.clip(node_homo_results + node_homo_std_results, 0, 1), alpha=0.7, color='orange')
    plt.fill_between(graph_hvalue_set, np.clip(edge_homo_results - edge_homo_std_results, 0, 1),
                     np.clip(edge_homo_results + edge_homo_std_results, 0, 1), alpha=0.7, color='blue')
    plt.fill_between(graph_hvalue_set, np.clip(class_homo_results - class_homo_std_results, 0, 1),
                     np.clip(class_homo_results + class_homo_std_results, 0, 1), alpha=0.7, color='pink')
    plt.fill_between(graph_hvalue_set, np.clip(agg_homo_results - agg_homo_std_results, 0, 1),
                     np.clip(agg_homo_results + agg_homo_std_results, 0, 1), alpha=0.7, color='purple')
    plt.fill_between(graph_hvalue_set, np.clip(ge_results - ge_std_results, 0, 1),
                     np.clip(ge_results + ge_std_results, 0, 1), alpha=0.7, color='yellow')
    plt.fill_between(graph_hvalue_set, np.clip(adj_homo_results - adj_homo_std_results, 0, 1),
                     np.clip(adj_homo_results + adj_homo_std_results, 0, 1), alpha=0.7, color='blue')
    plt.fill_between(graph_hvalue_set, np.clip(LI_results - LI_std_results, 0, 1),
                     np.clip(LI_results + LI_std_results, 0, 1), alpha=0.7, color='grey')

    plt.plot(graph_hvalue_set, 0.5 * np.ones(graph_hvalue_set.shape), '--', color='green')
    plt.plot(graph_hvalue_set, 0.05 * np.ones(graph_hvalue_set.shape), '--', color='orange')
    legends = ['GCN Performance', 'MLP-2 Performance', 'SGC Performance', 'MLP-1 Performance', 'node homo',
               'class homo', 'agg homo', 'GE homo', 'adjusted homo', 'LI', 'NT0.5']
    plt.legend(legends, bbox_to_anchor=(1, -0.02), loc='lower right')
    plt.title("Baseline Metrics on " + base_dataset)
    plt.xlabel('Edge Homophily')
    plt.xlim(0, 1)
    plt.ylim(-0.2, 1)

    fig_name = f"./plots/{base_dataset}_baseline_homo_performance_comparison.pdf"
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    plt.close()

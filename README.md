# When Do Graph Neural Networks Help with Node Classification?
Official repository for "When Do Graph Neural Networks Help with Node Classification? Investigating the Impact of Homophily Principle on Node Distinguishability" (Sitao Luan *et al.*, NeurIPS 2023) [paper](https://arxiv.org/abs/2304.14274)


Example of Node Distinguishability             | Classifier-based Performance Metrics and Baseline GNNs' Behavior on Synthetic Graphs
:-------------------------:|:-------------------------:
![](https://github.com/SitaoLuan/When-Do-GNNs-Help/blob/main/Figures%20in%20paper/example.png)  |  ![](https://github.com/SitaoLuan/When-Do-GNNs-Help/blob/main/Figures%20in%20paper/cora_synthetic_KR_nobar.png)

## Repository Overview
The repository is organised as follows:

```python
|-- csbm-h.py # experiments on 10 small-scale datasets
|-- homphily_tests.py # experiments on the large-scale and small-scale datasets based on the data provided by LINKX
|-- synthetic_plot.py # 3 old datasets, including cora, citeseer, and pubmed
|-- homophily.py # 6 new datasets, including texas, wisconsin, cornell, actor, squirrel, and chameleon
|-- utils.py # generate synthetic features and graphs with different homophily levels and train baseline models
|-- gnns_on_syn.py # all experimental plots and visualizations in our paper
```

## Dependencies

The script has been tested running under Python 3.7.4, with the following packages installed (along with their dependencies):

- `dgl-cpu==0.4.3.post2`
- `dgl-gpu==0.4.3.post2`
- `ogb==1.3.1`
- `numpy==1.19.2`
- `scipy==1.4.1`
- `networkx==2.5`
- `torch==1.5.0`
- `torch-cluster==1.5.7`
- `torch-geometric==1.6.3`
- `torch-scatter==2.0.5`
- `torch-sparse==0.6.6`
- `torch-spline-conv==1.2.0`

In addition, CUDA 10.2 has been used.

```
pip install -r requirements.txt
```

## Ablation Study on CSBM-H

### Play with CSBM-H

```
# Change the prior distributions with different n0, n1 settings
n0 = 100
n1 = 100
# Change the node centers for ablation on inter-class node distinguishability
mu0 = np.array([-1,0])
mu1 = np.array([1,0])
# Change sigma0 and sigma1 for ablation on intra-class node distinguishability
sigma0 = 1
sigma1 = 5
# Change d0, d1 for ablation on node degree
d0 = 5 
d1 = 5 
```
The results are saved into the `csbmh_plots` folder and the generated data will be saved at `csbmh_plots/data`.

## Performance Metrics on Real-world Datasets


Download the `data/` folder from the repository of [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) to `<your-working-directory>/data`.

```
# Run homophily_test.py to get the performance metrics
python train.py --homophily_test.py
```

## Tests on Synthetic Graphs
We use the data generation method from [ACM-GNNs](https://github.com/SitaoLuan/ACM-GNN), you can customize the synthetic graphs. We provide a set of generated graphs in 'data_synthesis'.

Generated features are saved into the `data_synthesis/features/` folder.

Generated graphs are saved into the `data_synthesis/<random|regular>/` folder.

We train and finetune SGC with 1 layer (sgc-1) and GCN on each synthetic graph, the results are store in 'gnns_on_syn.py'

### Plot the Results on Synthetic Graphs
Plot the results of baseline GNNs performance, the existing and proposed performance metrics as Appendix H.6 in the paper.

```
python synthetic_plot.py
```


## Attribution
Parts of the code are based on
- [ACM-GNNs](https://github.com/SitaoLuan/ACM-GNN)

## Reference
If you make advantage of this repository in your research, please cite the following in your manuscript:

```
@article{luan2023graph,
  title={When do graph neural networks help with node classification: Investigating the homophily principle on node distinguishability},
  author={Luan, Sitao and Hua, Chenqing and Xu, Minkai and Lu, Qincheng and Zhu, Jiaqi and Chang, Xiao-Wen and Fu, Jie and Leskovec, Jure and Precup, Doina},
  journal={arXiv preprint arXiv:2304.14274},
  year={2023}
}
```

## License
MIT


from __future__ import division
from __future__ import print_function
from pathlib import Path
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import scipy


# Results and std of GCN-rw
gcn_cora_mean = np.array([100,100,100,100,100,100,100,100,99.98,99.83,97.47,88.83,59.33,29.38,26.72,24.55,21.18,19.68,19.47,19.57,19.55,19.78,19.62,19.48,19.88,19.48,19.58,19.72,63.48,100])
gcn_cora_std = np.array([0,0,0,0,0,0,0,0,0.07,0.16,0.63,2.8,2.17,2.42,2.47,2.37,1.85,0.62,1.39,0.98,1.06,1.39,1.42,1.06,0.57,0.77,0.96,1.49,6.62,0])
gcn_citeseer_mean = np.array([100,100,100,100,100,100,100,100,99.98,99.65,97.42,88.33,57.25,26.55,24.2,21.1,19.7,19.6,19.35,19.88,19.73,19.48,19.6,19.65,19.5,19.45,19.85,19.62,49.5,98.47])
gcn_citeseer_std = np.array([0,0,0,0,0,0,0,0,0.07,0.41,0.99,2.56,2.74,1.88,2.04,2.36,1.51,1.39,1.34,2.03,0.98,0.78,0.85,0.97,1.17,0.99,1.03,0.98,7.29,1.02])
gcn_chameleon_mean = np.array([100,100,100,100,100,100,100,100,99.9,99.23,93.65,79.85,50.25,28.32,25.68,23.95,22.02,20.22,20.38,19.88,19.35,19.95,19.85,19.38,19.95,19.6,19.8,19.95,52.92,99.9])
gcn_chameleon_std = np.array([0,0,0,0,0,0,0,0,0.12,0.44,1.72,2.68,1.82,2.23,1.57,1.56,2.56,1.46,1.14,1.31,1.17,1.2,1.39,1.64,1.49,0.92,1.44,1.44,3.58,0.12])
gcn_film_mean = np.array([100,100,100,100,100,100,100,99.9,99.17,97.1,86.35,71.32,43.15,24.62,23.4,22.27,20.98,19.62,19.83,19.8,19.68,19.35,19.85,19.42,19.32,19.7,19.73,19.6,40.5,94.55])
gcn_film_std = np.array([0,0,0,0,0,0,0,0.12,0.42,0.76,1.98,0.97,1.45,2.43,2.48,2.29,2.19,1.75,1.27,1.36,2.06,1.1,1.15,1.32,1.34,1.01,1.17,0.87,2.53,3.99])
gcn_squirrel_mean = np.array([100,100,100,100,100,100,100,100,99.8,98.6,91.13,73.12,46.52,26.75,26.02,23.62,22.03,19.78,19.95,19.9,19.55,19.7,19.52,19.6,19.55,19.68,19.55,19.45,46.4,99.43])
gcn_squirrel_std = np.array([0,0,0,0,0,0,0,0,0.22,0.46,1.22,3.05,2.04,0.94,2.6,2.11,2.53,0.85,1.5,1.22,1.13,0.97,1.5,1.02,0.63,1.21,1.51,0.68,5.24,0.76])
gcn_random_mean = np.array([100,100,96.1,65.33,20.75,19.88,20.98,20.2,19.78,19.42,19.42,19.48,19.55,19.55,19.75,19.82,19.32,19.78,19.8,19.6,19.55,19.43,19.68,19.57,19.38,19.82,19.6,19.85,19.53,19.7])
gcn_random_std = np.array([0,0,11.7,33.87,6.31,5.1,3.26,1.72,2.31,0.91,0.78,1.02,1.48,0.78,0.84,0.83,1.02,1.74,0.86,1.38,1.03,0.32,1.06,1.06,1.28,1.08,0.87,1.36,0.79,1.51])
gcn_pubmed_mean = np.array([100,100,100,100,100,100,100,99.77,98.75,96.6,85.02,66.05,41.28,25.65,23.93,21.9,19.42,19.5,19.68,19.4,19.38,19.62,19.42,19.6,20,19.32,19.93,19.8,41.2,64.25])
gcn_pubmed_std = np.array([0,0,0,0,0,0,0,0.17,0.54,1.15,1.81,4.04,2.31,2.13,1.83,2.14,1.19,0.94,0.75,0.92,1.06,1.37,1.04,0.93,1.37,1.35,1.11,1.04,5.11,3.2])


gcn_mean=[gcn_cora_mean,gcn_citeseer_mean,gcn_pubmed_mean,gcn_chameleon_mean,gcn_squirrel_mean,gcn_film_mean] #,gcn_film_mean, gcn_dblp_mean, gcn_phy_mean, gcn_dblp_mean,gcn_cs_mean,gcn_phy_mean,gcn_comp_mean,gcn_photo_mean
gcn_std = [gcn_cora_std,gcn_citeseer_std,gcn_pubmed_std,gcn_chameleon_std,gcn_squirrel_std,gcn_film_std] #,gcn_film_std, gcn_dblp_std,gcn_phy_std, gcn_dblp_std,gcn_cs_std,gcn_phy_std,gcn_comp_std,gcn_photo_std
# dataset_name = ['cora','citeseer','pubmed','chameleon', 'squirrel', 'film'] #, 'film', 'CitationFull_dblp', 'Coauthor_Physics', 'Coauthor_CS','Amazon_Computers', 'Amazon_Photo', 'random'
gcn_dataset_name = ['cora_gcn','citeseer_gcn','pubmed_gcn','chameleon_gcn','squirrel_gcn','film_gcn'] #,'film_gcn','dblp_gcn','coauthor_phy_gcn','dblp_gcn','coauthor_cs_gcn','coauthor_phy_gcn','amz_comp_gcn','amz_photo_gcn'


mfgcn_cora_mean = np.array([100,100,100,100,100,100,100,100,100,99.85,99.15,97.18,92.7,84.78,85.2,84.17,83.52,82.88,82.97,83.22,83.32,83.55,83.25,84.28,84.53,85.28,85.95,88.28,99.12,100])
mfgcn_cora_std = np.array([0,0,0,0,0,0,0,0,0,0.12,0.56,0.83,0.95,2.2,1.91,1.36,1.43,2.42,1.55,1.76,1.46,1.46,1.67,1.9,1.63,2.06,1.41,1.21,0.83,0])
mfgcn_citeseer_mean = np.array([100,100,100,100,100,100,100,100,99.98,99.68,98.62,95.7,88.92,79.1,77.62,75.88,74.5,73.72,74.4,75.35,75.25,74.72,76.05,75.7,77.08,77.78,78.3,82.62,98.95,100])
mfgcn_citeseer_std = np.array([0,0,0,0,0,0,0,0,0.07,0.27,0.61,1.11,1.74,1.87,1.82,2.25,2.13,1.15,2.07,2.59,2.53,1.43,1.3,1.72,1.92,2.03,1.96,1.48,0.31,0])
mfgcn_chameleon_mean = np.array([100,100,100,100,100,100,100,100,99.7,99,92.85,83,65.45,49.4,49.9,47.97,47.25,46.7,48,48.02,49.82,50.1,50.57,52.35,53.25,54.45,56.05,60.43,91.38,99.98])
mfgcn_chameleon_std = np.array([0,0,0,0,0,0,0,0,0.51,0.62,1.42,2.11,4.62,3.57,2.4,2.35,3.03,1.87,2.07,2.87,1.59,2.49,2.17,2.42,1.92,1.78,1.75,2.75,2.98,0.07])
mfgcn_film_mean = np.array([100,100,100,100,100,100,100,99.5,97.65,93.3,77.58,64.38,47.35,34.2,34.33,33.9,34.35,33.88,35.18,35.23,35.88,35.1,34.85,35.7,35.63,36.1,38.15,40.1,73.3,95.62])
mfgcn_film_std = np.array([0,0,0,0,0,0,0,0.93,2.09,3.57,6.04,4.39,1.4,2.98,2.25,2.41,2.09,2.69,2.15,2.63,1.64,2.82,2.01,3.53,1.88,1.6,2.14,3.35,4.26,2.65])
mfgcn_squirrel_mean = np.array([100,100,100,100,100,100,100,99.98,98.93,96.1,85.33,63.82,44.88,30.12,29.33,28.3,28.6,29.7,30.27,31.52,31.67,32.45,33,33.7,35.5,35.83,37.62,40.62,78.92,99.95])
mfgcn_squirrel_std = np.array([0,0,0,0,0,0,0,0.07,1.71,0.45,3.41,7.7,3.32,2.01,2.48,1.89,2.34,1.8,2.1,2.35,1.84,2.29,1.8,2.55,2.73,2.02,1.2,2.67,8.66,0.1])
mfgcn_random_mean = np.array([96.95,96.43,97.85,90.78,78.38,65.85,57.05,62.45,57.98,61.68,55.9,49.65,20.58,19.35,19.7,19.93,19.5,20.15,19.53,20.1,20.45,21.22,21.25,22.42,23.15,23.77,24.3,26.93,80.7,100])
mfgcn_random_std = np.array([5.97,7.67,2.55,12.47,19.4,19.6,18.65,5.13,13.93,2.41,2.58,3.71,6.71,0.78,1.43,1.51,1.92,1.19,1.69,2.05,2.58,1.83,2.49,1.13,1.79,1.92,2.81,2.62,1.67,0])
mfgcn_pubmed_mean = np.array([100,100,100,100,100,100,99.98,99.5,97.15,94.1,82.93,74.08,61.52,52.33,50.75,50.48,50.08,50,50,50.52,50.8,50.68,51.08,51.75,51.45,52.03,52.58,55.5,81.23,99.63])
mfgcn_pubmed_std = np.array([0,0,0,0,0,0,0.07,0.46,1.28,1.26,3.79,4.02,3.03,2.12,2.92,1.72,2.11,1.53,3.02,2.09,1.61,2.79,2.06,2.45,2.15,1.46,3.09,2.66,3.71,0.28])

mfgcn_mean=[mfgcn_cora_mean, mfgcn_citeseer_mean, mfgcn_pubmed_mean, mfgcn_chameleon_mean, mfgcn_squirrel_mean, mfgcn_film_mean] #,mfgcn_film_mean, mfgcn_dblp_mean, mfgcn_phy_mean,mfgcn_dblp_mean,mfgcn_cs_mean,mfgcn_phy_mean,mfgcn_comp_mean,mfgcn_photo_mean
mfgcn_std = [mfgcn_cora_std, mfgcn_citeseer_std, mfgcn_pubmed_std, mfgcn_chameleon_std, mfgcn_squirrel_std, mfgcn_film_std] #, mfgcn_dblp_std,mfgcn_phy_std, mfgcn_dblp_std,mfgcn_cs_std,mfgcn_phy_std,mfgcn_comp_std,mfgcn_photo_std
# dataset_name = ['cora','citeseer','pubmed','chameleon', 'squirrel'] #, , 'film', 'CitationFull_dblp', 'Coauthor_Physics', 'Coauthor_CS','Amazon_Computers', 'Amazon_Photo', 'random'
mfgcn_dataset_name = ['cora_mfgcn','citeseer_mfgcn','pubmed_mfgcn','chameleon_mfgcn','squirrel_mfgcn', 'film_mfgcn'] #,'film_mfgcn','dblp_mfgcn','coauthor_phy_mfgcn','dblp_mfgcn','coauthor_cs_mfgcn','coauthor_phy_mfgcn','amz_comp_mfgcn','amz_photo_mfgcn'


sgc_cora_mean = np.array([99.75,99.85,99.63,99.23,98.95,98.6,98,95.28,93.12,88.77,82,73.1,58.72,38.88,37,32.73,28.48,24.92,21.75,20.7,20.78,19.75,20.55,21.05,21.43,22.27,25.02,30.15,70,97.98])
sgc_cora_std = np.array([0.19,0.17,0.28,0.38,0.27,0.34,0.43,0.77,1.47,1.74,1.7,1.7,2.41,2.15,2.3,2.07,2.54,2.86,1.37,1.85,2.33,1.69,1.54,2.38,2.05,1.14,2.04,1.95,2.02,0.85])
sgc_citeseer_mean = np.array([99.85,99.95,99.65,99.7,98.95,98.45,97.2,95.6,92.35,88.92,82.45,75.12,59.4,41.72,38.88,33.9,28.38,24.53,20.33,20.65,20.42,20.25,20.03,20.45,21.3,21.48,24.98,31.45,77.77,99.55])
sgc_citeseer_std = np.array([0.23,0.1,0.25,0.19,0.58,0.67,1.02,0.6,1.06,2.02,1.52,2.5,2.08,1.92,2.35,1.36,2.73,1.66,1.89,1.07,1.39,2,1.09,1.89,1.66,2.49,1.82,2.33,1.29,0.35])
sgc_chameleon_mean = np.array([97.1700,97.2300,95.7800,92.6000,89.7700,87.3000,83.2200,75.8500,68.0800,62.1800,54.0000,46.4500,37.4500,27.1200,26.6000,25.3000,22.3200,21.5800,21.3000,20.4000,20.4500,20.7800,20.2000,21.2200,20.35,20.95,21.78,23.4500,40.8200,82.5000])
sgc_chameleon_std = np.array([0.79,0.83,0.76,1.24,1.69,1.57,2.31,2.54,2.34,2.56,2.37,1.93,1.78,1.96,1.40,1.78,2.11,2.79,1.38,1.59,1.89,1.92,1.17,2.10,1.62,2.53,2.4,1.68,1.86,2.62])
sgc_film_mean = np.array([84.4500,83.83,80.70,77.07,72.15,69.80,66.25,60.83,54.60,48.02,43.05,37.68,30.93,24.97,23.80,23.02,21.50,21.73,20.85,20.40,20.25,20.38,19.82,20.50,20.45,20.15,21.9,21.70,33.25,52.73])
sgc_film_std = np.array([2.04,1.5,2.35,2.52,2.54,2.01,3.71,2.66,3.26,3.95,2.3,1.86,1.85,1.86,2.1,2.26,1.61,1.94,1.66,2.14,1.85,1.01,2.03,2.02,1.2,2.46,2.41,1.47,1.7,3.11])
sgc_squirrel_mean = np.array([95.77,95.75,92.9,89.1,86.03,82.55,78.72,72.23,65.33,57.52,49.62,41.9,33.3,25.53,24.6,23.03,21.85,20.67,20.5,20.67,19.9,20.73,21.33,20.38,20.75,20.62,21.4,22.6,40.82,81.92])
sgc_squirrel_std = np.array([0.91,1.22,1.03,1.3,1.82,1.86,1.25,2.16,3.1,2.04,2.02,2.09,3.13,1.79,1.53,1.02,2.52,2.12,2.27,2.76,1.42,1.13,3.21,1.77,2.21,2.46,2.14,2.21,1.55,2.32])
sgc_random_mean = np.array([91.03,92.6,83.67,76.98,71.7,61.68,57.43,49.88,41.33,35.48,26.18,25.65,21,19.57,19,19.43,19.57,19.48,19.32,19.8,19.42,19.65,19.3,19.22,19.32,19.73,19.45,19.48,19.78,21.43])
sgc_random_std = np.array([9.15,6.26,9.22,13.43,8.87,9.1,14.89,10.84,7.98,8.5,9.08,4.44,2.41,1.19,0.99,0.9,1.51,0.96,1.09,1.03,0.89,1.14,0.97,1.08,1.02,1.3,1.54,1.51,1.59,2.85])
sgc_pubmed_mean = np.array([91.78,92.45,90.98,89.23,87.53,85.8,85.25,81.68,76.05,72.88,64.5,57.45,45.45,31.95,30.53,29,27.18,24.2,21.35,20.67,20.3,19.8,20.4,20.03,21.65,23.68,23.53,26.85,54.33,85.43])
sgc_pubmed_std = np.array([1.66,1.26,1,1.96,2,1.4,1.99,1.95,1.93,2.11,1.81,1.96,2.47,2.25,2.41,2.12,2.18,2.15,2.17,2.1,1.82,2.69,2.06,1.81,1.69,2,1.92,2.24,1.73,1.57])

sgc_mean=[sgc_cora_mean, sgc_citeseer_mean, sgc_pubmed_mean, sgc_chameleon_mean, sgc_squirrel_mean, sgc_film_mean] #,sgc_film_mean, sgc_dblp_mean, sgc_phy_mean, sgc_dblp_mean,sgc_cs_mean,sgc_phy_mean,sgc_comp_mean,sgc_photo_mean
sgc_std = [sgc_cora_std,sgc_citeseer_std,sgc_pubmed_std,sgc_chameleon_std,sgc_squirrel_std,sgc_film_std] #,sgc_film_std, sgc_dblp_std,sgc_phy_std,sgc_dblp_std,sgc_cs_std,sgc_phy_std,sgc_comp_std,sgc_photo_std
# dataset_name = ['cora','citeseer','pubmed','chameleon', 'squirrel'] #, , 'film', 'CitationFull_dblp', 'Coauthor_Physics','Coauthor_CS','Amazon_Computers', 'Amazon_Photo', 'random'
sgc_dataset_name = ['cora_sgc','citeseer_sgc','pubmed_sgc','chameleon_sgc','squirrel_sgc','film_sgc'] #,'film_sgc','dblp_sgc','coauthor_phy_sgc','dblp_sgc','coauthor_cs_sgc','coauthor_phy_sgc','amz_comp_sgc','amz_photo_sgc'


mfsgc_cora_mean = np.array([100,100,100,99.98,100,99.92,99.85,99.33,98.75,96.8,92.4,88.1,83.6,83.78,83.6,83.8,83.1,83.3,83.55,83.4,83.75,83.5,83.95,84.38,84.88,84.6,84.73,86.23,89.2,91.48])
mfsgc_cora_std = np.array([0,0,0,0.07,0,0.11,0.2,0.5,0.51,1.04,1.25,1.76,1.16,1.87,1.49,1.72,1.12,1.78,1.86,1.64,1.88,1.18,1.43,1.32,1.71,2.57,2.05,0.88,0.96,0.93])
mfsgc_citeseer_mean = np.array([100,100,100,100,99.9,99.8,99.52,98.38,74.6,76.13,74.35,74.92,74.78,76.15,75.02,75.17,74.55,75.83,74.65,74.35,75.17,75.17,76.97,75.65,76.77,76.72,77.7,77.83,82.3,85.55])
mfsgc_citeseer_std = np.array([0,0,0,0,0.17,0.19,0.47,0.39,2.45,1.88,2.06,2.11,1.85,2.17,2.34,1.52,1.58,2.34,2.19,2.2,2.26,1.63,1.59,1.73,3.06,2.16,2.57,1.79,1.68,1.52])
mfsgc_chameleon_mean = np.array([99.9800,99.9800,99.8000,99.6500,99.3300,98.2000,96.6500,87.9300,72.5800,64.4500,56.2300,54.3300,48.5200,47.7000,48.1000,47.3800,47.4500,48.1500,48.9300,48.5800,49.4700,49.0000,48.9500,51.8300,51.4800,51.8200,52.3700,55.7000,62.1000,71.3300])
mfsgc_chameleon_std = np.array([0.07,0.07,0.22,0.28,0.46,0.93,0.91,1.96,3.33,3.37,2.40,2.25,2.31,2.44,3.13,1.93,2.11,2.01,2.27,2.08,1.98,2.27,1.52,2.55,1.57,2.03,2.80,1.71,2.14,2.53])
mfsgc_film_mean = np.array([98.4500,98.83,97.17,95.28,93.45,89.95,87.45,79.68,72.10,64.18,53.85,46.85,36.93,33.55,33.25,33.08,33.50,33.30,34.48,33.90,34.00,35.30,33.9800,35.15,35.05,35.52,35.00,36.90,45.45,73.52])
mfsgc_film_std = np.array([0.66,0.43,0.78,1.29,1.14,1.6,1.88,2.04,2.56,2.6,1.93,3.03,1.89,1.35,1.31,2.24,2.95,2.5,1.22,1.74,2.51,3.21,1.64,1.57,1.87,2.06,1.74,2.14,2.68,4.72])
mfsgc_squirrel_mean = np.array([99.75,99.88,99.2,98.32,97.25,94.9,91.58,80.13,67.3,56.75,43.57,36.33,30.47,27.47,26.8,27.52,25.7,27.7,26.67,28.93,28.02,28.02,29.30,29.68,28.82,30.53,32.45,32.92,39.03,48.02])
mfsgc_squirrel_std = np.array([0.3,0.17,0.31,0.6,0.84,0.75,1.42,2.46,1.63,2.72,1.49,2.1,2.78,3.04,3.06,1.72,3.4,1.74,3.35,1.22,2.18,2.05,2.13,1.94,2.17,1.54,1.77,1.22,2.92,2.35])
mfsgc_random_mean = np.array([26.05,22.65,22.12,20.2,20.98,20.85,20.5,20.33,21.35,21.37,20.53,20.73,20.5,20.65,21.05,21.48,20.02,20.65,20.53,20.55,21.43,20.93,21.15,22.15,22.55,23.3,22.88,24.13,29.72,34.98])
mfsgc_random_std = np.array([5.41,4.55,4.99,1.35,1.4,1.73,1.83,1.78,1.98,2.35,2.07,1.14,1.49,1.55,2,2.08,1.93,2.04,1.48,1.19,2.01,2.03,2.4,2.02,1.09,2.94,3.31,1.17,2.02,1.37])
mfsgc_pubmed_mean = np.array([96.2,95.88,94.68,93.83,91.83,91.53,88.27,82.95,71.85,64.22,59.32,54.75,50.58,50.08,50.48,50.5,50.28,50.15,50.15,50.17,50.55,50.63,51.13,51.38,51.23,52.62,52.2,53.12,57.2,59.5])
mfsgc_pubmed_std = np.array([0.77,0.91,1.06,1.5,0.45,1.4,1.44,2.26,3.5,2.71,2.57,2.25,2.87,2.31,2.72,3.31,1.78,2.24,2.76,3.22,3.58,3.16,2.3,3.25,2.17,1.35,2.15,1.9,2.83,1.55])

mfsgc_mean=[mfsgc_cora_mean,mfsgc_citeseer_mean,mfsgc_pubmed_mean,mfsgc_chameleon_mean,mfsgc_squirrel_mean,mfsgc_film_mean] #,mfsgc_film_mean, mfsgc_dblp_mean, mfsgc_phy_mean, mfsgc_dblp_mean,mfsgc_cs_mean,mfsgc_phy_mean,mfsgc_comp_mean,mfsgc_photo_mean
mfsgc_std = [mfsgc_cora_std,mfsgc_citeseer_std,mfsgc_pubmed_std,mfsgc_chameleon_std,mfsgc_squirrel_std,mfsgc_film_std] #,mfsgc_film_std, mfsgc_dblp_std,mfsgc_phy_std, mfsgc_dblp_std,mfsgc_cs_std,mfsgc_phy_std,mfsgc_comp_std,mfsgc_photo_std
mfsgc_dataset_name = ['cora_mfsgc','citeseer_mfsgc','pubmed_mfsgc','chameleon_mfsgc','squirrel_mfsgc','film_mfsgc'] #,'film_mfsgc','dblp_mfsgc','coauthor_phy_mfsgc','dblp_mfsgc','coauthor_cs_mfsgc','coauthor_phy_mfsgc','amz_comp_mfsgc','amz_photo_mfsgc'

graph_svalue_set = [0.05, 0.1, 0.15, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
mlp1_cora_mean = np.repeat(81.55, len(graph_svalue_set))
mlp1_cora_std = np.repeat(1.73, len(graph_svalue_set))
mlp1_citeseer_mean = np.repeat(73.25, len(graph_svalue_set))
mlp1_citeseer_std = np.repeat(2.07, len(graph_svalue_set))
mlp1_chameleon_mean = np.repeat(48.5, len(graph_svalue_set))
mlp1_chameleon_std = np.repeat(1.43, len(graph_svalue_set))
mlp1_film_mean = np.repeat(32, len(graph_svalue_set))
mlp1_film_std = np.repeat(2.36, len(graph_svalue_set))
mlp1_squirrel_mean = np.repeat(24.8, len(graph_svalue_set))
mlp1_squirrel_std = np.repeat(2.83, len(graph_svalue_set))
mlp1_random_mean = np.repeat(20.95, len(graph_svalue_set))
mlp1_random_std = np.repeat(1.3, len(graph_svalue_set))
mlp1_pubmed_mean = np.repeat(47.4, len(graph_svalue_set))
mlp1_pubmed_std = np.repeat(2.49, len(graph_svalue_set))

mlp1_mean=[mlp1_cora_mean, mlp1_citeseer_mean, mlp1_pubmed_mean, mlp1_chameleon_mean, mlp1_squirrel_mean, mlp1_film_mean] #gcn_dblp_mean,gcn_cs_mean,gcn_phy_mean,gcn_comp_mean,gcn_photo_mean
mlp1_std = [mlp1_cora_std, mlp1_citeseer_std, mlp1_pubmed_std, mlp1_chameleon_std, mlp1_squirrel_std, mlp1_film_std] #gcn_dblp_std,gcn_cs_std,gcn_phy_std,gcn_comp_std,gcn_photo_std
mlp1_dataset_name = ['cora_mlp1','citeseer_mlp1','pubmed_mlp1','chameleon_mlp1','squirrel_mlp1','film_mlp1'] #,'dblp_gcn','coauthor_cs_gcn','coauthor_phy_gcn','amz_comp_gcn','amz_photo_gcn'


mlp2_cora_mean = np.repeat( 83.75, len(graph_svalue_set))
mlp2_cora_std = np.repeat( 0.95, len(graph_svalue_set))
mlp2_citeseer_mean = np.repeat(74.8 , len(graph_svalue_set))
mlp2_citeseer_std = np.repeat( 1.56, len(graph_svalue_set))
mlp2_chameleon_mean = np.repeat(47.9 , len(graph_svalue_set))
mlp2_chameleon_std = np.repeat(2.31 , len(graph_svalue_set))
mlp2_film_mean = np.repeat(33.8 , len(graph_svalue_set))
mlp2_film_std = np.repeat( 3.25, len(graph_svalue_set))
mlp2_squirrel_mean = np.repeat(28.55 , len(graph_svalue_set))
mlp2_squirrel_std = np.repeat(2.2 , len(graph_svalue_set))
mlp2_random_mean = np.repeat(20.05 , len(graph_svalue_set))
mlp2_random_std = np.repeat(1.93 , len(graph_svalue_set))
mlp2_pubmed_mean = np.repeat(50.1 , len(graph_svalue_set))
mlp2_pubmed_std = np.repeat(2.47 , len(graph_svalue_set))

mlp2_mean=[mlp2_cora_mean, mlp2_citeseer_mean,mlp2_pubmed_mean, mlp2_chameleon_mean, mlp2_squirrel_mean, mlp2_film_mean] #gcn_dblp_mean,gcn_cs_mean,gcn_phy_mean,gcn_comp_mean,gcn_photo_mean, mlp2_dblp_mean, mlp2_phy_mean
mlp2_std = [mlp2_cora_std, mlp2_citeseer_std,mlp2_pubmed_std, mlp2_chameleon_std, mlp2_squirrel_std, mlp2_film_std] #gcn_dblp_std,gcn_cs_std,gcn_phy_std,gcn_comp_std,gcn_photo_std, mlp2_dblp_std, mlp2_phy_std
mlp2_dataset_name = ['cora_mlp2','citeseer_mlp2','pubmed_mlp2','chameleon_mlp2','squirrel_mlp2','film_mlp2'] #,'dblp_gcn','coauthor_cs_gcn','coauthor_phy_gcn','amz_comp_gcn','amz_photo_gcn','dblp_mlp2','coauthor_phy_mlp2'

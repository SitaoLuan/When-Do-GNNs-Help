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
from torch_scatter import scatter_add

from utils import random_disassortative_splits, accuracy

pi = math.pi
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
device = torch.device(device)



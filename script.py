import numpy as np
from time import time
import random, math
import networkx as nx

from logit_bpmf import LNMF
from pmf import PMF


# # read .gml
# G = nx.read_gml('data/dolphins-v62-e159/dolphins.gml') # 62 vertices
# A = nx.adjacency_matrix(G).todense()   #(62，62)   
# pmf = PMF(A, train_size=0.7)
# mat = pmf.read_triplets(file='edge.txt')
# pmf.train()


"""
PMF
"""
A = PMF.read_triplets(file='edge.txt')
pmf = PMF(A, train_size=0.7)
pmf.train()

"""
Bayesian NMF
"""

# A = LNMF.read_triplets(file='edge.txt')
# lnmf = LNMF(A, r=2)


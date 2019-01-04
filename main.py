import numpy as np
from time import time
import random, math
import networkx as nx

from logit_bpmf import LNMF
from pmf import PMF



""" read .gml & initialize triplets data form """
# G = nx.read_gml('data/dolphins-v62-e159/dolphins.gml') # 62 vertices
# mat = nx.adjacency_matrix(G).todense()   #(62，62)   
# PMF.convert_triplets('test_case1.txt', mat)


r=10

"""  PMF  """
A = PMF.read_triplets(file='test_case1.txt')
pmf = PMF(A,num_feature=r, train_size=0.7)
t1 = time()
pmf.train()
t2 = time()
print("Initialization finished.\nTime taken:", t2-t1)
error = pmf.validation()
print("Current error is: ", error)

w0 = pmf.get_w1_W1()
h0 = pmf.get_w1_H1()

"""  Bayesian NMF  """
lnmf = LNMF(A, r=r, w1_W1=w0, w1_H1=h0, max_iter = 1)
lnmf.train()




# mat = pmf.read_triplets(file='edge.txt')
# pmf.train()


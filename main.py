import numpy as np
from scipy.stats import multivariate_normal
from time import time
import random, math
import networkx as nx

from logit_bpmf import LNMF
from pmf import PMF

"""Importing data."""
G_full = nx.Graph()
G_test = nx.Graph()
print("Importing data...")
t0 = time()
c = 0
with open('data/facebook/facebook_full.txt','r') as f:
	count = 0
	# with open('data/facebook/missing_terms.txt','a') as ff:
	for line in f:
		line=line.split()#split the line up into a list - the first entry will be the node, the others his friends
		count = count + 1
		if count%5 != 0:
			c = c+1
			focal_node = line[0]#pick your node
			for friend in line[1:]:#loop over the friends
				if friend != focal_node:
					G_full.add_edge(focal_node,friend)#add each edge to the graph
					G_test.add_edge(focal_node,friend)
		else:
			focal_node = line[0]#pick your node
			for friend in line[1:]:#loop over the friends
				if friend != focal_node:
					G_full.add_edge(focal_node,friend)#add each edge to the graph
			if line[0] not in G_test:
				G_test.add_node(line[0])
			if line[1] not in G_test:
				G_test.add_node(line[1])
					# ff.write('{} {}\n'.format(line[0], line[1]))
f.close()
# ff.close()

if(c>0):
	print(c) # mod 5, 70588, 80%

A_full = nx.adjacency_matrix(G_full).todense()   #(4039, 4039)
A_test = nx.adjacency_matrix(G_test).todense()
t1 = time()
print("Importing finished. The test matrix has shape:", A_test.shape, "\n")

# """ read .gml & initialize triplets data form """
# G = nx.read_gml('data/dolphins-v62-e159/dolphins.gml') # 62 vertices
# mat = nx.adjacency_matrix(G).todense()   #(62，62)
# matsize = mat.shape[0]
# # PMF.convert_triplets('test_case1.txt', mat)


r=10

""" PMF """
""" Will run it at the final test"""
# A = PMF.read_triplets(file='data/facebook/pmf_test_case1.txt')
# pmf = PMF(A, A_test, Asize=A_test.shape[0], num_feature=r, maxepoch = 1, train_size=0.7, epsilon=50, reg = 0.01, momentum=0.8)
# t1 = time()
# pmf.train()
# t2 = time()
# print("Initialization of w_0 and h_0 finished.\nTime taken:", t2-t1)
# error = pmf.validation()
# print("Current error is: ", error)

# w0 = pmf.get_w1_W1()
# h0 = pmf.get_w1_H1()
# print("w0: ", w0)
# print("h0: ", h0)
# print("w0 and h0 has shape: ", w0.shape, h0.shape, "\n")
""""""

"""  Bayesian NMF  """
w0, h0 = np.zeros((A_test.shape[0], r)),  np.zeros((A_test.shape[0], r))
A_test_upper = np.triu(A_test)
# mat_lower = np.tril(mat)
lnmf = LNMF(A_test_upper, r=r, w1_W1=w0, w1_H1=h0, max_iter = 100)
lnmf.mh_train()


count = 0
correct = 0
with open('data/facebook/missing_terms.txt','r') as ff:
	
	# with open('data/facebook/missing_terms.txt','a') as ff:
	for line in f:
		count = count + 1
		line=line.split()#split the line up into a list - the first entry will be the node, the others his friends
		if lnmf.predict(line[0], line[1]) == 1:
			correct = correct + 1

print("Predict [", count, "] entries has correct numbers: [", correct ,"]\nCorrect prediction ratio is:", correct/count)
ff.close()

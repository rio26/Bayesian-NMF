import numpy as np
from time import time
import random, math
import networkx as nx

class PMF():
	def __init__(self, A, train_size):
		epsilon = 50 # Learning rate
		reg = 0.01 # Regularization parameter
		momentum=0.8; 
		epoch=1; 
		maxepoch=50

		# train - 70%; test - 30% 
		self.Asize = A.shape[0]
		self.mean_aij = np.sum(A) / (self.Asize * self.Asize)
		self.train_sample_size = math.ceil(self.Asize * train_size)
		self.test_sample_size = self.Asize - self.train_sample_size

		# initialize train and test sample.
		self.train_sample = np.copy(A[0:self.train_sample_size,0:self.train_sample_size])
		self.test_sample = np.copy(A[self.train_sample_size:, self.train_sample_size:])

		self.numbatches= 9; # Number of batches  
		self.num_feature = 10; # Rank 10 decomposition 
		self.w1_W1     = 0.1* np.random.standard_normal(size = (self.Asize, self.num_feature)) # W's feature vector
		self.w1_H1     = 0.1* np.random.standard_normal(size = (self.Asize, self.num_feature)) # H's feature vecators
		self.w1_W1_inc = np.zeros((self.Asize, self.num_feature))
		self.w1_H1_inc = np.zeros((self.Asize, self.num_feature))

	def convert_triplets(self, file):	
		"""======= create a Triplets: {w_id, h_id, binary_link} ======="""
		# with open('edge.txt', 'a') as f:
		with open(file, 'a') as f:
			for i in range(self.Asize):
				for j in range(i, self.Asize):
					if A[i,j] == 1:
						f.write('{} {} 1\n'.format(i, j))
		f.close()

	def read_triplets(self, file):
		tri_matrix = np.loadtxt(file, dtype='i', delimiter=' ')
		return tri_matrix

	def train(self):
		for epoch in range(self.maxepoch):
			rr = np.random.permutation(self.train_sample_size)
			train_vector = self.train_sample[rr,:]
			for batch in range(self.numbatches):
				N = math.ceil(self.Asize/self.numbatches) # number training triplets per batch 
				# print(N)

# read .gml
G = nx.read_gml('data/dolphins-v62-e159/dolphins.gml') # 62 vertices
A = nx.adjacency_matrix(G).todense()   #(62，62)   
pmf = PMF(A, train_size=0.7)
mat = pmf.read_triplets(file='edge.txt')
print(type(mat))
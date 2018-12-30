import numpy as np
from time import time
import random, math
import networkx as nx

class PMF():
	def __init__(self, A, train_size, max_iter=100, epsilon=50, reg = 0.01, momentum=0.8, maxepoch = 50):
		self.epsilon = epsilon # Learning rate
		self.reg = reg # Regularization parameter
		self.momentum=momentum 
		self.maxepoch = maxepoch
		self.error_trend = np.zeros((max_iter))

		# train - 70%; test - 30% 
		self.Asize = A.shape[0]
		self.mean_aij = np.sum(A[:,2]) / self.Asize
		
		self.train_sample_size = math.ceil(self.Asize * train_size)
		self.test_sample_size = self.Asize - self.train_sample_size

		# initialize train and test sample.
		self.train_sample = np.copy(A[0:self.train_sample_size,0:self.train_sample_size])
		self.test_sample = np.copy(A[self.train_sample_size:,:])

		self.numbatches= 5; # Number of batches  
		self.num_feature = 10; # Rank 10 decomposition 
		self.w1_W1     = 0.1* np.random.standard_normal(size = (self.Asize, self.num_feature)) # W's feature vector
		self.w1_H1     = 0.1* np.random.standard_normal(size = (self.Asize, self.num_feature)) # H's feature vecators
		self.w1_W1_inc = np.zeros((self.Asize, self.num_feature))
		self.w1_H1_inc = np.zeros((self.Asize, self.num_feature))

	def train(self):
		for epoch in range(self.maxepoch):
			rr = np.random.permutation(self.train_sample_size)
			train_vector = self.train_sample[rr,:]
			for batch in range(1, self.numbatches-1):
				N = math.ceil(self.Asize/self.numbatches) # number training triplets per batch
				aa_w = train_vector[((batch-1)*N):batch*N, 0]
				aa_h = train_vector[((batch-1)*N):batch*N, 1]
				aa = train_vector[((batch-1)*N):batch*N, 2]
				aa = aa - self.mean_aij   # Default prediction is the mean of adjacency matrix A. 

				''' Compute Prediction'''
				pred_out = np.multiply(self.w1_W1[aa_w,:],self.w1_H1[aa_h,:]).sum(axis=1)
				f = sum( np.power((pred_out - aa),2) + 0.5 * self.reg *((np.power(self.w1_W1[aa_w,:],2) + np.power(self.w1_H1[aa_h,:],2)).sum(axis=1)))

				''' Compute Gradient'''
				IO = np.tile(2*np.array([pred_out - aa]).T, (1, self.num_feature))
				Ix_w = np.multiply(IO, self.w1_H1[aa_h,:]) + self.reg*self.w1_W1[aa_w,:]
				Ix_h = np.multiply(IO,self.w1_W1[aa_w,:]) + self.reg*self.w1_H1[aa_h,:]
				dw1_W1 = np.zeros((self.Asize,self.num_feature))
				dw1_H1 = np.zeros((self.Asize,self.num_feature))

				for ii in range(N-1):
					dw1_W1[aa_w[ii],:] =  dw1_W1[aa_w[ii],:] +  Ix_w[ii,:]
					dw1_H1[aa_h[ii],:] =  dw1_H1[aa_h[ii],:] +  Ix_h[ii,:]

				''' Update movie and user features'''
				self.w1_W1_inc = self.momentum* self.w1_W1_inc + self.epsilon*dw1_W1/N
				self.w1_W1 = self.w1_W1 - self.w1_W1_inc

				self.w1_H1_inc = self.momentum* self.w1_H1_inc + self.epsilon*dw1_H1/N
				self.w1_H1 = self.w1_H1 - self.w1_H1_inc

				''' Compute Prediction after Parameter Updates'''
				pred_out = np.multiply(self.w1_W1[aa_w,:],self.w1_H1[aa_h,:]).sum(axis=1)
				f_s = sum( np.power((pred_out - aa),2) + 0.5 * self.reg *((np.power(self.w1_W1[aa_w,:],2) + np.power(self.w1_H1[aa_h,:],2)).sum(axis=1)))

				# print(f_s)
	
	def validation(self):
		# print(self.test_sample)
		aa_w = self.test_sample[:,0]
		aa_h = self.test_sample[:,1]
		aa_A = self.test_sample[:,2]

		pred = np.multiply(self.w1_H1[aa_h,:], self.w1_W1[aa_w,:]).sum(axis=1) + self.mean_aij
		print(pred.shape)
		error = math.sqrt(((pred - aa_A)*(pred - aa_A)).sum()/ aa_A.shape[0] )
		return error



	def get_w1_W1(self):
		return self.w1_W1

	def get_w1_H1(self):
		return self.w1_H1		

	def convert_triplets(file):	
		"""======= create a Triplets: {w_id, h_id, binary_link} ======="""
		# with open('edge.txt', 'a') as f:
		with open(file, 'a') as f:
			for i in range(self.Asize):
				for j in range(i, self.Asize):
					if A[i,j] == 1:
						f.write('{} {} 1\n'.format(i, j))
		f.close()

	def read_triplets(file):
		tri_matrix = np.loadtxt(file, dtype='i', delimiter=' ')
		return tri_matrix

# # # read .gml
# # G = nx.read_gml('data/dolphins-v62-e159/dolphins.gml') # 62 vertices
# # A = nx.adjacency_matrix(G).todense()   #(62，62)   
# A = read_triplets("edge.txt")
# pmf = PMF(A, train_size=0.7)
# pmf.train()
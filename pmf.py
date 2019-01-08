import numpy as np
from time import time
import random, math
import networkx as nx

class PMF():
	def __init__(self, A, mat, Asize, train_size, numbatches =5, num_feature=5, max_iter=100, epsilon=50, reg = 0.01, momentum=0.8, maxepoch = 10):
		self.epsilon = epsilon # Learning rate
		self.reg = reg # Regularization parameter
		self.momentum=momentum 
		self.maxepoch = maxepoch
		self.error_trend = np.zeros((max_iter))

		# train - 70%; test - 30% 
		# self.Asize = Asize
		self.Asize = A.shape[0]
		self.wsize = mat.shape[0]
		self.hsize = mat.shape[1]
		self.mean_aij = np.sum(A[:,2]) / A.size
		
		self.train_sample_size = math.ceil(self.Asize * train_size)
		self.test_sample_size = self.Asize - self.train_sample_size
		# print("train and test sample size", self.train_sample_size, self.test_sample_size)


		# initialize train and test sample.
		# self.train_sample = np.asmatrix(np.copy(mat[0:self.train_sample_size,0:self.train_sample_size])) # (44,44)
		# # print("trainsample:", self.train_sample.shape)
		# self.test_sample = np.asmatrix(np.copy(mat[self.train_sample_size:,self.train_sample_size:])) # (18,18)
		# # print("testsample", self.test_sample.shape)

		self.train_sample = np.copy(A[0:self.train_sample_size,:])
		# print("trainsample:", self.train_sample.shape)
		self.test_sample = np.copy(A[self.train_sample_size:,:])
		# print("testsample", self.test_sample.shape)

		self.numbatches= numbatches; # Number of batches  
		self.num_feature = num_feature; # Rank of decomposition 
		self.w1_W1     = 0.1* np.random.standard_normal(size = (self.wsize, num_feature)) # W's feature vector (n,r)
		# print("w1_W1.shape",self.w1_W1.shape)
		self.w1_H1     = 0.1* np.random.standard_normal(size = (self.hsize, num_feature)) # H's feature vecators (m,r)
		# print("w1_H1.shape",self.w1_H1.shape)
		self.w1_W1_inc = np.zeros((self.wsize, self.num_feature))  # (n,r)
		self.w1_H1_inc = np.zeros((self.hsize, self.num_feature))  # (n,r)
		# print("w1_H1_inc.shape",self.w1_H1_inc.shape)

	def train(self):
		for epoch in range(self.maxepoch):
			# rr = np.random.permutation(self.train_sample_size)
			# print("rr", rr.shape) # (44,)
			# train_vector = self.train_sample[rr,:] # (44,)
			# print("train_vector", type(self.train_sample))
			np.random.shuffle(self.train_sample) #(44,44)
			train_vector = self.train_sample
			# print("train_vector", train_vector.shape)

			for batch in range(1, self.numbatches-1):
				N = math.ceil(self.Asize/self.numbatches) # number training triplets per batch
				aa_w = np.array(train_vector[((batch-1)*N):batch*N, 0]) # (N,)

				aa_h = train_vector[((batch-1)*N):batch*N, 1] # (N,)
				aa_len = np.array(range(0, self.num_feature)) # (r,)
				aa = train_vector[((batch-1)*N):batch*N, 2]   # (N,)
				aa = aa - self.mean_aij   # (N,) Default prediction is the mean of adjacency matrix A. 

				a = self.w1_W1[np.ix_(aa_w, aa_len)] # (N,r)
				b = self.w1_H1[np.ix_(aa_h, aa_len)] # (N,r)

				pred_out = np.multiply(a,b).sum(axis=1) # (N,)

				f2= (np.power(a,2) + np.power(b,2)).sum(axis=1)
				f = (np.power(pred_out-aa,2) +0.5*self.reg*(f2)).sum()
				# print("fffff" , f)

				''' Compute Gradient'''
				IO = np.tile(2*np.array([pred_out - aa]).T, (1, self.num_feature))
				Ix_w = np.multiply(IO,a) + self.reg*a
				if batch < 1:
					print(Ix_w)
				Ix_h = np.multiply(IO,b) + self.reg*b
				dw1_W1 = np.zeros((self.wsize,self.num_feature))
				dw1_H1 = np.zeros((self.hsize,self.num_feature))
				# print("dw1_W1 & dw1_H1" ,dw1_H1.shape, "hellow", self.num_feature)

				for ii in range(N-1):
					dw1_W1[aa_w[ii],:] =  dw1_W1[aa_w[ii],:] +  Ix_w[ii,:]
					dw1_H1[aa_h[ii],:] =  dw1_H1[aa_h[ii],:] +  Ix_h[ii,:]

				# print(dw1_W1)
				''' Update movie and user features'''
				self.w1_W1_inc = self.momentum* self.w1_W1_inc + self.epsilon*dw1_W1/N
				self.w1_W1 = self.w1_W1 - self.w1_W1_inc

				self.w1_H1_inc = self.momentum* self.w1_H1_inc + self.epsilon*dw1_H1/N
				self.w1_H1 = self.w1_H1 - self.w1_H1_inc

				''' Compute Prediction after Parameter Updates'''
				# pred_out = np.multiply(self.w1_W1[aa_w,:],self.w1_H1[aa_h,:]).sum(axis=1)
				# f_s = sum( np.power((pred_out - aa),2) + 0.5 * self.reg *((np.power(self.w1_W1[aa_w,:],2) + np.power(self.w1_H1[aa_h,:],2)).sum(axis=1)))

				a = self.w1_W1[np.ix_(aa_w, aa_len)] # (N,r)
				b = self.w1_H1[np.ix_(aa_h, aa_len)] # (N,r)
				pred_out = np.multiply(a,b).sum(axis=1) # (N,)

				f2_s= (np.power(a,2) + np.power(b,2)).sum(axis=1)
				fs = (np.power(pred_out-aa,2) +0.5*self.reg*(f2_s)).sum()
				# print("fffff" , fs)

				# print(f_s)
	
	def validation(self):
		# print(self.test_sample)
		aa_w = self.test_sample[:,0]
		aa_h = self.test_sample[:,1]
		aa_A = self.test_sample[:,2]

		pred = np.multiply(self.w1_H1[aa_h,:], self.w1_W1[aa_w,:]).sum(axis=1) + self.mean_aij
		error = math.sqrt(((pred - aa_A)*(pred - aa_A)).sum()/ aa_A.shape[0] )
		return error

	def get_w1_W1(self):
		return self.w1_W1

	def get_w1_H1(self):
		return self.w1_H1		

	def convert_triplets(output_file, mat):	
		"""======= create a Triplets: {w_id, h_id, binary_link} ======="""
		# with open('edge.txt', 'a') as f:
		with open(output_file, 'a') as f:
			for i in range(mat.shape[0]):
				for j in range(i, mat.shape[0]):
					if mat[i,j] == 1:
						f.write('{} {} 1\n'.format(i, j))
					else:
						f.write('{} {} 0\n'.format(i, j))
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
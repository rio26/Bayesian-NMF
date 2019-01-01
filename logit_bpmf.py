import numpy as np
import random
import pandas as pd
from numpy.random import multivariate_normal
from scipy.stats import wishart
from numpy.linalg import inv
from time import time

class LNMF():
	"""
	A - adjancency matrix
	r - number of features
	"""
	def __init__(self, A, r, w1_W1, w1_H1, maxepoch = 50, max_iter = 100):
		# self.epsilon = epsilon # Learning rate
		print("Initializing...")
		self.maxepoch = maxepoch
		self.Asize = A.shape[0]
		self.r = r
		self.max_iter = max_iter
		self.mean_a = np.sum(A[:,2]) / self.Asize
		self.error_trend = np.zeros((max_iter))

		""" Initialize the hierarchical priors: """
		self.beta =2 # observation noise (precision)
		self.mu_w = np.zeros((r, 1))
		self.mu_h = np.zeros((r, 1))
		self.alpha_w = np.eye(r)
		self.alpha_h = np.eye(r)
		
		""" 
	    Parameters for Inverse-Wishart distribution
    	Assuming that parameters for both U and V are the same.
    	"""
		self.WI_w = np.eye(r)
		self.WI_w_inv = inv(self.WI_w)   # rxr
		self.b0_w = 2
		self.df_w = r
		self.mu0_w = np.zeros((r, 1))

		self.WI_h = np.eye(r)
		self.WI_h_inv = inv(self.WI_h)
		self.b0_h = 2
		self.df_h = r
		self.mu0_h = np.zeros((r, 1))

		"""
		Initialization Bayesian PMF using MAP solution found by PMF
		"""
		self.w1_W1_sample = w1_W1.T
		# print("input sample size", self.w1_W1_sample.shape)
		self.w1_H1_sample = w1_H1.T
		# print(self.w1_H1_sample.shape)
		self.mu_w = np.array([w1_W1.mean(0)]).T
		alpha_w = inv(np.cov(w1_W1))
		
		self.mu_h = np.array([w1_H1.mean(0)]).T
		alpha_h = inv(np.cov(w1_H1))
		print("LNMF Initialization done.")

		# self.mu_0 = np.zeros(r)
		# self.nu_0 = r
		# self.Beta_0 = 2
		# self.W_0 = np.eye(r)

	def train(self):
		print("Training...")
		N_w = self.w1_W1_sample.shape[0]
		# print("line 67", N)
		iteration = self.max_iter
		for i in range(iteration):
			""" Sample hyperparameter conditioned on the 
    		current column and row features."""
			w_bar = self.w1_W1_sample.mean(axis=0)  # (n, 1)
			# print("w_bar's shape", w_bar.shape)
			w_cov = np.cov(self.w1_W1_sample) # (r,r)
			# print(w_cov.shape)
			WI_post = inv(self.WI_w_inv + N_w*w_cov +\
				(self.b0_w*N_w*np.dot(self.mu0_w-w_bar,(self.mu0_w-w_bar).T))/(self.b0_w+N_w))
			print("WI_post", WI_post.shape)
			# Beta_0_star = self.Beta_0 + self.Asize
			# nu_0_star = self.nu_0 + self.Asize
			# W_0_inv = np.linalg.inv(W_0) # compute the inverse once and for all


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
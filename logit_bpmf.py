import numpy as np
import random
import pandas as pd
from numpy.random import multivariate_normal
from scipy.stats import wishart
from utilities import Gaussian_Wishart, gaussian_error
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
		self.gaussian_errors = gaussian_error(10000, 1) ## check sigma later
		self.error_trend = np.zeros((max_iter))

		""" Initialize the hierarchical priors: """
		# self.mu_w = np.zeros((r, 1))
		# self.mu_h = np.zeros((r, 1))
		# self.alpha_w = np.eye(r)
		# self.alpha_h = np.eye(r)
		
		""" 
	    Parameters for Inverse-Wishart distribution
    	Assuming that parameters for both U and V are the same.
    	"""
		self.WI_w = np.eye(r)
		self.WI_w_inv = inv(self.WI_w)   	# rxr
		self.mu0_w = np.zeros((r, 1))

		self.WI_h = np.eye(r)
		self.WI_h_inv = inv(self.WI_h)   	# rxr
		self.mu0_h = np.zeros((r, 1))

		self.b0 = 2							# observation noise (precision)
		df = r 								# degree of freedom
		self.beta = self.b0 + self.r 		# fixed
		self.nu = df + self.r 				# fixed 

		"""
		Initialization Bayesian PMF using MAP solution found by PMF
		"""
		self.w1_W1_sample = w1_W1.T
		self.w1_H1_sample = w1_H1.T
		print("Input sample size, W: ", self.w1_W1_sample.shape,\
			"\nInput sample size, H: ", self.w1_H1_sample.shape)

		self.mu_w = np.array([w1_W1.mean(0)]).T
		self.alpha_w = np.eye(r)
		# self.alpha_w = inv(np.cov(w1_W1))
		
		self.mu_h = np.array([w1_H1.mean(0)]).T
		self.alpha_h = np.eye(r)
		# self.alpha_h = inv(np.cov(w1_H1))
 
		print("LNMF Initialization done.")

	def train(self):
		print("\nTraining...")
		N_w = self.w1_W1_sample.shape[0]
		N_h = self.w1_H1_sample.shape[0]
		# print(N_w, N_h)

		# nu_0_star = self.nu_0 + self.Asize
		# W_0_inv = np.linalg.inv(W_0) # compute the inverse once and for all
		# print("line 67", N)
		iteration = self.max_iter
		for ite in range(iteration):
			""" Sample hyperparameter conditioned on the current COLUMN features."""
			w_bar = self.w1_W1_sample.mean(axis=1).reshape((-1,1))  # (n, 1)
			w_cov = np.cov(self.w1_W1_sample) # (r,r)
			WI_post = self.compute_wishart0(mat=self.WI_w_inv, n=N_w, cov=w_cov,mu0=self.mu0_w,s_bar=w_bar)
			mu_tmp = ((self.b0*self.mu0_w + N_w*w_bar)/(self.b0+N_w)).reshape((-1,))	# [self.b0+N_w] can be substituded to [self.beta] 
			self.mu_w, self.alpha_w, lamd_w = Gaussian_Wishart(mu_tmp, self.beta, WI_post, self.nu, seed=None)


			""" Sample hyperparameter conditioned on the current ROW features."""
			h_bar = self.w1_H1_sample.mean(axis=1).reshape((-1,1))  # (n, 1)
			h_cov = np.cov(self.w1_H1_sample) # (r,r)
			WI_post = self.compute_wishart0(mat=self.WI_h_inv ,n =N_h, cov=h_cov, mu0=self.mu0_h, s_bar=h_bar)	
			mu_tmp = ((self.b0*self.mu0_w + N_h*h_bar)/(self.b0+N_w)).reshape((-1,))	# [self.b0+N_w] can be substituded to [self.beta] 
			self.mu_h, self.alpha_h, lamd_h = Gaussian_Wishart(mu_tmp, self.beta, WI_post, self.nu, seed=None)

			for gibbs in range(2):
				### This can be done by multi-threads to speed up if Asize is large. ###
				# for i in range(self.Asize):  # Sample W
				for i in range(2):
					# MCMC for finding the mean of logit normal
					tmp_w = self.w1_W1_sample[:,i].reshape((-1,1)).T  # (1,r)
					# print("tmp_w",tmp_w.shape)
					for j in range(self.Asize):
						tmp_h = self.w1_H1_sample[:,j].reshape((-1,1)) # (r,1)
						tmp_mu =  np.dot(tmp_w,tmp_h) # (1,1)
						Y = tmp_mu + self.gaussian_errors # (10000,1)
						logit_y = self.logistic(Y) # (10000,1)
						l_mean = np.mean(logit_y)
						print("Y", l_mean)
					return

	def compute_wishart0(self, mat, n, cov, mu0,s_bar):
		wi = inv(mat + n*cov + (self.b0*n*np.dot(mu0-s_bar,(mu0-s_bar).T))/(self.b0+n))
		print("computed and obtained size", wi.shape)
		return (wi+wi.T)/2

	def logit_nomral_mean():
		return

	def logistic(self,x):
		return 1.0 / (1.0 + np.exp(-x))

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
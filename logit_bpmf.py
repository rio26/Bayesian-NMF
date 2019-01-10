import numpy as np
import random
from numpy.random import multivariate_normal
from scipy.stats import wishart
from scipy.stats import multivariate_normal as mul_normal
from utilities import Gaussian_Wishart, gaussian_error
from numpy.linalg import inv
from time import time

class LNMF():
	"""
	A - adjancency matrix
	r - number of features
	"""
	def __init__(self, A, mat, r, w1_W1, w1_H1, maxepoch = 50, max_iter = 100):
		# self.epsilon = epsilon # Learning rate
		print("Initializing...")
		self.maxepoch = maxepoch
		self.Asize = A.shape[0]
		self.wsize = mat.shape[0]
		self.hsize = mat.shape[1]

		self.r = r
		self.max_iter = max_iter
		self.mean_a = np.sum(A[:,2]) / self.Asize
		self.gaussian_errors = gaussian_error(10000, 1) ## check sigma later
		self.error_trend = np.zeros((max_iter))
		
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
		self.w1_W1_sample = w1_W1.T # (r, n)
		self.w1_H1_sample = w1_H1.T # (r, n)
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

		iteration = self.max_iter
		for ite in range(iteration):
			""" Sample hyperparameter conditioned on the current COLUMN features."""
			w_bar = self.w1_W1_sample.mean(axis=1).reshape((-1,1))  # (n, 1)
			w_cov = np.cov(self.w1_W1_sample) # (r,r)
			WI_post = self.compute_wishart0(mat=self.WI_w_inv, n=N_w, cov=w_cov,mu0=self.mu0_w,s_bar=w_bar)
			mu_tmp = ((self.b0*self.mu0_w + N_w*w_bar)/(self.b0+N_w)).reshape((-1,))	# [self.b0+N_w] can be substituded to [self.beta] 
			self.mu_w, self.alpha_w, lamd_w = Gaussian_Wishart(mu_tmp, self.beta, WI_post, self.nu, seed=None)  # mean, matrix, covariance


			""" Sample hyperparameter conditioned on the current ROW features."""
			h_bar = self.w1_H1_sample.mean(axis=1).reshape((-1,1))  # (n, 1)
			h_cov = np.cov(self.w1_H1_sample) # (r,r)
			WI_post = self.compute_wishart0(mat=self.WI_h_inv ,n =N_h, cov=h_cov, mu0=self.mu0_h, s_bar=h_bar)	
			mu_tmp = ((self.b0*self.mu0_w + N_h*h_bar)/(self.b0+N_w)).reshape((-1,))	# [self.b0+N_w] can be substituded to [self.beta] 
			self.mu_h, self.alpha_h, lamd_h = Gaussian_Wishart(mu_tmp, self.beta, WI_post, self.nu, seed=None)  # mean (5,), matrix, covariance
			print("size:", self.alpha_h.shape)

			for gibbs in range(2):
				### This can be done by multi-threads to speed up if Asize is large. ###
				for i in range(self.wsize):  # Sample W
				# for i in range(2):
					tmp_w = self.w1_W1_sample[:,i].reshape((-1,1)).T  # (1,r)
					tmp_mean = 1
					wi_pdf = mul_normal.pdf(tmp_w.T, self.mu_w, lamd_w) #(r,)
					# print(wi_pdf)
					
					for j in range(self.hsize):
						mean_j = self.logit_nomral_mean(a=tmp_w, b=self.w1_H1_sample[:,j].reshape((-1,1)), error=self.gaussian_errors)
						tmp_mean = tmp_mean * mean_j
						# tmp_mean = tmp_mean + mean_j
						# print(j)
						# print(mean_j)
					print(tmp_mean * wi_pdf)
				for j in range(self.hsize):  # Sample W
				# for i in range(2):
					tmp_w = self.w1_W1_sample[:,i].reshape((-1,1)).T  # (1,r)
					tmp_mean = 0
					wi_pdf = mul_normal.pdf(tmp_w.T, self.mu_w, lamd_w) #(r,)
					
					for j in range(self.hsize):
						mean_j = self.logit_nomral_mean(a=tmp_w, b=self.w1_H1_sample[:,j].reshape((-1,1)), error=self.gaussian_errors)
	

# np.random.binomial(size=3, n=1, p= 0.5)

	def compute_wishart0(self, mat, n, cov, mu0,s_bar):
		wi = inv(mat + n*cov + (self.b0*n*np.dot(mu0-s_bar,(mu0-s_bar).T))/(self.b0+n))
		print("computed and obtained size", wi.shape)
		return (wi+wi.T)/2

	# MC for finding the mean of logit normal
	def logit_nomral_mean(self, a,b, error):
		mu = np.dot(a,b) 	# a (1,r); b (r,1)
		Y = mu + error
		# Y = mu + gaussian_error(10000, 1)
		logit_y = self.logistic(Y) # (10000,1)
		mean = np.mean(logit_y)
		return mean

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
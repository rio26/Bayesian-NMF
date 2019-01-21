import numpy as np
import random
from numpy.random import multivariate_normal
from scipy.stats import wishart
from scipy.stats import multivariate_normal as mul_normal
from utilities import Gaussian_Wishart, gaussian_error
from numpy.linalg import inv
from time import time
import warnings
# import multiprocessing
# from multiprocessing import Pool

class Dirichlet_Cat():
	"""
	A - adjancency matrix
	r - number of features
	"""
	def __init__(self, mat, r, w1_W1, w1_H1, maxepoch = 50, max_iter = 100, burnin = 0):
		# self.epsilon = epsilon # Learning rate
		print("Initializing LNMF...")
		# self.core = multiprocessing.cpu_count()
		# self.pool = Pool(processes=multiprocessing.cpu_count()) 
		self.maxepoch = maxepoch
		self.wsize = mat.shape[0]
		self.hsize = mat.shape[1]
		self.burnin = burnin

		self.mat = mat
		self.r = r
		self.max_iter = max_iter
		self.gaussian_errors = gaussian_error(sigma=1, num=10000) ## check sigma later
		# self.error_trend = np.zeros((int(max_iter/10)))
		self.error_trend = np.zeros((max_iter))

		"""
	    Parameters for Inverse-Wishart distribution
    	Assuming that parameters for both U and V are the same.
    	"""
		self.WI_w = np.eye(r, dtype='float64')
		self.WI_w_inv = inv(self.WI_w)   	# rxr
		self.mu0_w = np.zeros((r, 1))

		self.WI_h = np.eye(r, dtype='float64')
		self.WI_h_inv = inv(self.WI_h)   	# rxr
		self.mu0_h = np.zeros((r, 1))

		self.b0 = 2							# observation noise (precision)
		df = r 								# degree of freedom
		self.beta = self.b0 + self.r 		# fixed
		self.nu = df + self.r 				# fixed

		"""
		Initialization latent features using MAP solution found by PMF
		"""
		self.w1_W1_sample = w1_W1.T # (r, n)
		self.w1_H1_sample = w1_H1.T # (r, n)
		# print("Input sample size, W: ", self.w1_W1_sample.shape,\
		# 	"\nInput sample size, H: ", self.w1_H1_sample.shape)

		self.mu_w = np.array([w1_W1.mean(0)]).T
		self.alpha_w = np.eye(r)
		# self.alpha_w = inv(np.cov(w1_W1))

		self.mu_h = np.array([w1_H1.mean(0)]).T
		self.alpha_h = np.eye(r)
		# self.alpha_h = inv(np.cov(w1_H1))

		self.pred_matrix = np.copy(mat)

		print("LNMF Initialization done.")

	def mh_train(self):
		print("\nTraining...")
		N_w = self.w1_W1_sample.shape[0]
		N_h = self.w1_H1_sample.shape[0]
		# print(N_w, N_h)

		iteration = self.max_iter
		posterior_w_old = np.float64(0)
		posterior_h_old = np.float64(0)

		posterior_w_cand = np.float64(0)
		posterior_h_cand = np.float64(0)

		for ite in range(iteration):
			print("[running iteration ", ite , "...]")
			# if (ite % 10 == 0):
			# 	self.error_trend[ite%10] = self.predict_accuracy()
			# 	print("Current error:", self.error_trend[ite%10])
			self.error_trend[ite] = self.predict_accuracy()
			print(self.error_trend[ite])
			with open('result.txt','a') as f:
				f.write('Iteration {} has accuracy {}\n'.format(ite, self.error_trend[ite]))
			f.close()
			t0 = time()
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


			"""MCMC for generating latent features"""
			with warnings.catch_warnings():
				warnings.filterwarnings('error')
				posterior_w_old =  self.metropolis_hasting(mcmc_iter=15, component_start=0, component_end=self.wsize, 
					lamd = lamd_w, component=self.w1_W1_sample, inner_loop_size=self.hsize, fixed_component=self.w1_H1_sample, 
					posterior_old=posterior_w_old, posterior_cand=posterior_w_cand, sigma=0.5)

				posterior_h_old =  self.metropolis_hasting(mcmc_iter=15, component_start=0, component_end=self.hsize, 
					lamd = lamd_h, component=self.w1_H1_sample, inner_loop_size=self.wsize, fixed_component=self.w1_W1_sample, 
					posterior_old=posterior_h_old, posterior_cand=posterior_h_cand, sigma=0.5)
			t1 = time()
			print("Iteration ", ite, " takes:", t1-t0)

	def metropolis_hasting(self, mcmc_iter, component_start, component_end, lamd, component, inner_loop_size, fixed_component, posterior_old, posterior_cand, sigma):
		for i in range(component_start, component_end):
			t3 = time()
			update_num = 0
			warn = 0
			for mc in range(mcmc_iter):
				if mc == 0:
					tmp_com = component[:,i].reshape((-1,1)).T # (1,r)
					tmp_likelihood = 0
					try: # if posterior isn't -inf
						prior = np.log(mul_normal.pdf(tmp_com, self.mu_w, lamd)) # num
						### Compute likelihood ###
						for j in range(inner_loop_size):
							mean_j = self.logit_nomral_mean(a=tmp_com, b=fixed_component[:,j].reshape((-1,1)), error=self.gaussian_errors)
							if(self.mat[i,j] == 1):
								tmp_likelihood = tmp_likelihood + np.log(mean_j)
							else:
								tmp_likelihood = tmp_likelihood + np.log(1-mean_j)

						posterior_old= prior + tmp_likelihood
					except Warning: # if posterior is -inf
						warn = warn + 1
						posterior_old = 0
				else:
					tmp_com = (component[:,i] + gaussian_error(sigma=sigma)).reshape((-1,1)).T # (1,r)
					tmp_likelihood = 0		
					try: # if new posterior isn't -inf
						prior = np.log(mul_normal.pdf(tmp_com, self.mu_w, lamd)) # num
							### Compute likelihood ###
						for j in range(inner_loop_size):
							mean_j = self.logit_nomral_mean(a=tmp_com, b=fixed_component[:,j].reshape((-1,1)), error=self.gaussian_errors)
							if(self.mat[i,j] == 1):
								tmp_likelihood = tmp_likelihood + np.log(mean_j)
							else:
								tmp_likelihood = tmp_likelihood + np.log(1-mean_j)
						# print("tmp_likelihood:", tmp_likelihood, "\n-------------------------------------")
						posterior_cand = tmp_likelihood + prior
						if posterior_old == 0: # if old posterior is -inf
							posterior_old = posterior_cand
							update_num = update_num + 1
							component[:,i] = tmp_com.T.reshape((-1,))
						else: # if old posterior isn't -inf
							if  np.random.uniform() < min(1, np.power(10, posterior_cand/posterior_old)):
								posterior_old = posterior_cand
								update_num = update_num + 1
								component[:,i] = tmp_com.T.reshape((-1,))
					except Warning: # if new posterior is -inf
						warn = warn + 1
						continue
			print("Update_num and Total MCMC steps for column with warnings", i, "is[", update_num, mcmc_iter, warn,"]")
			print("Column ", i, " takes:", time()-t3)
		return posterior_old

	def compute_wishart0(self, mat, n, cov, mu0,s_bar):
		wi = inv(mat + n*cov + (self.b0*n*np.dot(mu0-s_bar,(mu0-s_bar).T))/(self.b0+n))
		# print("computed and obtained wishart's size", wi.shape)
		return (wi+wi.T)/2 # wi and wi.T should be the same, this is used to garantee symmetricy.

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

	def predict(self,i,j):
		a = self.w1_W1_sample[:,i].reshape((-1,1)).T
		b = self.w1_H1_sample[:,j]
		p = np.dot(a,b)
		# print(a.shape, b.shape )
		# print(p.shape)
		return np.random.binomial(1, p)

	def predict_accuracy(self):
		count = 0
		correct = 0
		with open('data/facebook/missing_terms.txt','r') as ff:
			# with open('data/facebook/missing_terms.txt','a') as ff:
			for line in ff:
				count = count + 1
				line=line.split()#split the line up into a list - the first entry will be the node, the others his friends
				# print("type checking:", line[0], type(line[1]))
				if self.predict(int(line[0]), int(line[1])) == 1:
					correct = correct + 1
		ratio = correct/count
		print("Predict [", count, "] entries has correct numbers: [", correct ,"]\nCorrect prediction ratio is:", correct/count)
		ff.close()
		return ratio

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
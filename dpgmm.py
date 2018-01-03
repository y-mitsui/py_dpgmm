from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, wishart
from scipy.misc import factorial
import pandas as pd
import math
import sys

def RisingFact(x, n):
    if (n <= 1):
        return x
    else:
        return x * RisingFact(x + 1, n - 1)

class DPGMM:
    def __init__(self, covar, n_iter = 2000, u0=None,  n_cluster=2, alpha=1., beta=0.3, new=10.):
        self.n_cluster = n_cluster
        self.u0 = u0
        self.alpha = alpha
        self.beta = beta
        self.new = new
        self.n_iter = n_iter
    
    def getSampleK(self, k, latent_z,):
        sample_idx = np.where(latent_z == k)[0]
        return self.sample[sample_idx]
    
    def updateParameters(self, latent_z, n_cluster_samples):
        params = []
        normal_insts = []
        for k in range(self.n_cluster):
            sample_idx = np.where(latent_z == k)[0]
            sample_k = sample[sample_idx]
            n_sample_k = sample_k.shape[0]
            mean_sample_k = np.average(sample_k, 0)
            cov_k = np.zeros((n_dimentions, n_dimentions))
            for j in range(n_sample_k):
                deviation = sample_k[j] - mean_sample_k
                cov_k += np.dot(deviation.reshape(-1, 1), deviation.reshape(1, -1))
            
            deviation = mean_sample_k - self.u0
            tmp1 = np.dot(deviation.reshape(-1, 1), deviation.reshape(1, -1))
            tmp2 = (n_cluster_samples[k] * self.beta) / (n_cluster_samples[k] + self.beta)
            tmp3 = tmp2 * tmp1
            
            wish_cover = np.linalg.inv(np.linalg.inv(covar) + cov_k + tmp3)
            
            normal_cov = wishart.rvs(self.new + n_cluster_samples[k], wish_cover)
            tmp4 = (n_cluster_samples[k] + self.beta) * normal_cov
            tmp5 = n_cluster_samples[k] + self.beta
            tmp6 = (n_cluster_samples[k] * mean_sample_k + self.beta * self.u0) / tmp5
            normal_mean = multivariate_normal(tmp6, np.linalg.inv(tmp4)).rvs()
            params.append((normal_mean, np.linalg.inv(normal_cov)))
            normal_insts.append(multivariate_normal(normal_mean, np.linalg.inv(normal_cov)))
        return params, normal_insts
    
    def fit(self, sample):
        self.sample = np.array(sample)
        n_sample = self.sample.shape[0]
        n_dimentions = self.sample.shape[1]
        if self.u0 is None:
            self.u0 = np.average(self.sample, 0)
            
        latent_z = np.random.randint(0, self.n_cluster, size=n_sample)
        n_cluster_samples = pd.value_counts(latent_z).to_dict()
        
        params = []
        normal_insts = []
        for k in range(self.n_cluster):
            sample_k = self.getSampleK(k, latent_z)
            tmp_covar = np.cov(sample_k, rowvar=0)
            tmp_mean = np.average(sample_k, 0)
            params.append((tmp_mean, tmp_covar))
            normal_insts.append(multivariate_normal(params[-1][0], params[-1][1]))
        
        new_cluster_probs = []
        for j in range(n_sample):
            deviation = self.sample[j] - self.u0
            S_b_inv_1 = np.dot(deviation.reshape(-1, 1), deviation.reshape(1, -1))
            S_b_inv_2 = np.linalg.inv(covar)
            S_b_inv = S_b_inv_2 + (self.beta / (1 + self.beta)) * S_b_inv_1
            S_b_det = np.linalg.det(np.linalg.inv(S_b_inv))

            tmp1 = (self.beta / ((1 + self.beta) * np.pi)) ** (n_dimentions / 2.)
            tmp2 = (self.new + 1.) / 2.
            tmp3 = S_b_det ** tmp2 * math.gamma(tmp2)
            tmp4 = np.linalg.det(covar) ** (self.new / 2)
            tmp5 = tmp3 / (tmp4 * math.gamma((self.new + 1. - n_dimentions) / 2))
            p_x_theta = tmp1 * tmp5
            tmp6 = self.alpha / (n_sample - 1. + self.alpha)
            new_cluster_probs.append(tmp6 * p_x_theta)
        
        for i in range(self.n_iter):
            for j in range(n_sample):
                n_cluster_samples[latent_z[j]] -= 1
                
                if n_cluster_samples[latent_z[j]] == 0:
                    self.n_cluster -= 1
                    for k in range(n_sample):
                        if latent_z[k] > latent_z[j]:
                            latent_z[k] -= 1
                    params.pop(latent_z[j])
                    normal_insts.pop(latent_z[j])
                    
                    del n_cluster_samples[latent_z[j]]
                    new_n_cluster_samples = {}
                    for k, v in n_cluster_samples.items():
                        if k > latent_z[j]:
                            k -= 1
                        new_n_cluster_samples[k] = v
                    n_cluster_samples = new_n_cluster_samples
                
                probs_k = []
                for k in range(self.n_cluster):
                    tmp = n_cluster_samples[k] / (n_sample - 1 + self.alpha)
                    probs_k.append(normal_insts[k].pdf(self.sample[j]) * tmp)
                probs_k.append(new_cluster_probs[j])
                
                probs_k = np.array(probs_k) / np.sum(probs_k)
                prob_cum = np.cumsum(probs_k)
                val = np.random.uniform()
                param_idx = np.where(prob_cum >= val)[0][0]
                
                latent_z[j] = param_idx
                if param_idx in n_cluster_samples:
                    n_cluster_samples[param_idx] += 1
                else:
                    n_cluster_samples[param_idx] = 1
            self.n_cluster = max(latent_z) + 1
            
            params, normal_insts = self.updateParameters(latent_z, n_cluster_samples)

            if i % 100 == 0:
                print("n_cluster", self.n_cluster)
                for param in params:
                    print(" mean", param[0])
                    print(" cov", param[1])
                
np.random.seed(12345)

n_sample = 500
n_dimentions = 2
prob = [0.33, 0.33, 0.34]
means = [0, 4.5, 9.0]

prob_cum = np.cumsum(prob)
sample = []
for _ in range(n_sample):
    val = np.random.uniform()
    param_idx = np.where(prob_cum > val)[0][0]
    sample_mean = [means[param_idx]] * n_dimentions
    sample_std = np.eye(n_dimentions)
    sample.append(np.random.multivariate_normal(sample_mean, sample_std))
sample = np.array(sample)

plt.scatter(sample[:,0], sample[:,1])
plt.show()

covar = np.array([[1. , 0.],[0., 1.]])

dpgmm = DPGMM(covar, n_iter=1000, alpha=0.1, beta=.3)
dpgmm.fit(sample)


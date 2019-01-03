import filterpy.kalman as fp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


m = 1000
n = 1
z = 1_000_000
alpha = 1e-3
beta = 2.0
kappa = 0.0
means = np.linspace(-2.0, 2.0, m)
sigma = 1.14
points = fp.MerweScaledSigmaPoints(n, alpha, beta, kappa)
ut = np.empty_like(means)
sampling = np.empty_like(means)

for i, mean in enumerate(means):
    sigmas = points.sigma_points(mean, sigma**2)
    trans_sigmas = sigmoid(sigmas)
    ut[i], _ = fp.unscented_transform(trans_sigmas, points.Wm, points.Wc)

    x = np.random.normal(mean, sigma, z)
    sampling[i] = np.mean(sigmoid(x))

print(pd.DataFrame({"mu": means,
                    "sigma": sigma,
                    "ut": ut,
                    "sampling": sampling}))

x = np.random.normal(means[0], sigma, z)
plt.hist(sigmoid(x), bins=50)
plt.title("mu = {}, sigma = {}".format(means[0], sigma))
plt.xlabel("f(x)")
plt.show()
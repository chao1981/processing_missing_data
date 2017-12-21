
from scipy.stats import multivariate_normal as mvn
def em_gmm(xs, pis, mus, sigmas, tol=0.01, max_iter=100):

    n, p = xs.shape
    k = len(pis)

    ll_old = 0
    for _ in range(max_iter):
        exp_A = []
        exp_B = []
        ll_new = 0

        # E-step
        ws = np.zeros((k, n))
        for j in range(k):
            for i in range(n):
                density =  pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
                ws[j, i] = density
        ws /= ws.sum(0)

        # M-step
        pis = np.zeros(k)
        for j in range(k):
            for i in range(n):
                pis[j] += ws[j, i]
        pis /= n

        mus = np.zeros((k, p))
        for j in range(k):
            for i in range(n):
                mus[j] += ws[j, i] * xs[i]
            mus[j] /= ws[j, :].sum()

        sigmas = np.zeros((k, p, p))
        for j in range(k):
            for i in range(n):
                ys = np.reshape(xs[i]- mus[j], (4,1))
                sigmas[j] += ws[j, i] * np.dot(ys, ys.T)
            sigmas[j] /= ws[j,:].sum()

        # update complete log likelihoood
        ll_new = 0.0
        for i in range(n):
            s = 0
            for j in range(k):
                s += pis[j] * mvn(mus[j], sigmas[j]).pdf(xs[i])
            ll_new += np.log(s)

        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new

    return ws, pis, mus, sigmas

def init_weights(df):
    pass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel("totalNPdata.xlsx",  header=None)
idx_gmm = list(range(2,6))
df_gmm = df[idx_gmm]
df_gmm.shape

#Initial the parameters
n = 218
_shape = (5, df_gmm.shape[1])
_mus = np.random.random(_shape)
_sigmas = np.random.random(_shape)
_pis = np.random.random(_shape[0])
_pis = _pis / np.sum(_pis)
xs = np.array(df_gmm)

ws, pis, mus, sigmas = em_gmm(np.array(df_gmm), _pis, _mus, _sigmas, tol=0.01, max_iter=100)
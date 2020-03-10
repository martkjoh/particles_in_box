import numpy as np
from numpy import pi, sqrt

R = 5
a = 1
eps = 0.5
K = 10

def dot(xi, xj):
    return np.einsum("...jk, ...jk -> ...k", xi, xj)

def V_w(x):
    r = sqrt(dot(x[0], x[0]))
    indx = r>R
    E = np.zeros_like(r)
    E[indx] = K/2 * (r[indx] - R)**2
    return E

def V_i(k, x):
    x_kj = np.zeros_like(x[0])
    N = len(x[0, 0])
    for i in range(N):
        x_kj[:, i] = x[0, :, k] - x[0, :, i]
    r_kj = sqrt(dot(x_kj, x_kj))
    indx = r_kj<a
    indx[k] = False
    ar = a/r_kj[indx]
    E = eps*(ar**12 - 2*ar**6 + 1)
    return np.sum(E)

def E_k(x):
    return 1/2*dot(x[1], x[1])

def totalV_i(x):
    energy = np.zeros_like(x[0, 0])
    for k in range(len(energy)):
        energy[k] += 1/2 * V_i(k, x[0])
    return energy

def F_w(x0):
    r = sqrt(dot(x0, x0))
    indx = r>R
    F = np.zeros_like(x0)
    F[:, indx] = -K*(r[indx] - R) * x0[:, indx] / r[indx]
    return F

def F_i(k, x0):
    x_kj = np.zeros_like(x0)
    N = len(x0[0])
    for i in range(N):
        x_kj[:, i] = x0[:, k] - x0[:, i]
    r_kj = sqrt(dot(x_kj, x_kj))
    indx = r_kj<a
    indx[k] = False
    ar = a/r_kj[indx]
    F_kj = 12*eps*(ar**12 - ar**6)*x_kj[:, indx]/r_kj[indx]**2
    return np.einsum("ik -> i", F_kj)

def sumF(x0):
    f1 = F_w(x0)
    N = len(x0[0])
    for k in range(N):
        f1[:, k] += F_i(k, x0)
    return f1
import numpy as np
from numpy import pi, cos, sin, sqrt
from matplotlib import pyplot as plt
from matplotlib import cm
from time import time

R = 5
K = 10
N = 5
a = 1
eps = 0.5
timeSteps = 100000
dt = 0.01

def dot(xi, xj):
    return np.einsum("...jk, ...jk -> ...k", xi, xj)

def V_w(x0):
    r = sqrt(dot(x0, x0))
    indx = r>R
    E = np.zeros_like(r)
    E[indx] = K/2 * (r[indx] - R)**2
    return E

def V_i(k, x0):
    x_kj = np.zeros_like(x0)
    for i in range(N):
        x_kj[:, i] = x0[:, k] - x0[:, i]
    r_kj = sqrt(dot(x_kj, x_kj))
    indx = r_kj<a
    indx[k] = False
    E = eps*((a/r_kj[indx])**12 - 2*(a/r_kj[indx])**6 + 1)
    return np.sum(E)

def F_w(x0):
    r = sqrt(dot(x0, x0))
    indx = r>R
    F = np.zeros_like(x0)
    F[:, indx] = -K*(r[indx] - R) * x0[:, indx] / r[indx]
    return F

def F_i(k, x0):
    x_kj = np.zeros_like(x0)
    for i in range(N):
        x_kj[:, i] = x0[:, k] - x0[:, i]
    r_kj = sqrt(dot(x_kj, x_kj))
    indx = r_kj<a
    indx[k] = False
    absF_kj = 24*eps*((a/r_kj[indx])**12 - (a/r_kj[indx])**6)
    F_kj = absF_kj * x_kj[:, indx]/r_kj[indx]**2

    return np.einsum("ik -> i", F_kj)

def sumF(x0):
    f1 = F_w(x0)
    for k in range(N):
        f1[:, k] += F_i(k, x0)
    return f1

def verletStep(x, F):
    f1 = F(x[0])
    x0 = x[0] + x[1]*dt + 1/2*f1*dt**2
    x1 = x[1] + (f1 + F(x0)) / 2. * dt 
    return np.array([x0, x1])

# Returns the total energy of the system xt
def Et(xt):
    E = V_w(xt[:, 0]) + 1/2*dot(xt[:, 1], xt[:, 1])
    for t in range(timeSteps):
        for k in range(N):
            E[t, k] += V_i(k, xt[t, 0])
    return np.einsum("ij -> i", E)

# Places N particles randomly, but not ontop of echother, with speeds
# x[i, j, k] is i'th derivative of coordinate j of particle k
def getEmpty():
    print("placing particles...")
    xt0 = np.empty((2, 2, N))
    xt0[0, :, 0] = np.random.uniform(-R, R, 2)
    k = 1
    while k<N:
        xt0k = np.random.uniform(-R, R, 2)
        for j in range(k):
            r = xt0[0, :, j] - xt0k
            if r@r>(a)**2:
                pass
            else:
                break
            if j == k-1:
                xt0[0, :, k] = xt0k
                k += 1

    angles = np.random.uniform(-pi, pi, N)
    xt0[1] = np.array([cos(angles), sin(angles)])
    xt = np.zeros((timeSteps, *np.shape(xt0)))
    xt[0] = xt0
    print("Done!")
    return xt

xt = getEmpty()

t = time()
for i in range(timeSteps-1):
    xt[i+1] = verletStep(xt[i], sumF)
print(time() - t)

theta = np.linspace(0, 2*pi, 100)
X, Y = R*cos(theta), R*sin(theta)
plt.plot(X, Y)
for i in range(N):
    plt.plot(xt[0, 0, 0, i], xt[0, 0, 1, i], "rx")
    plt.plot(xt[:, 0, 0, i], xt[:, 0, 1, i], color = cm.viridis(i/N), alpha = 0.1)
plt.show()


# energy =  Et(xt)
# relEnergy = (energy - energy[0]) / energy[0]
# plt.plot(np.linspace(0, dt*timeSteps, timeSteps), relEnergy)
# plt.show()
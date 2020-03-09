import numpy as np
from numpy import pi, cos, sin,sqrt
from matplotlib import pyplot as plt
from matplotlib import cm
from time import time

R = 5
K = 10
N = 1
timeSteps = 10000
dt = 0.02

def dot(xi, xj):
    return np.einsum("...jk, ...jk -> ...k", xi, xj)

def V(x0):
    r = sqrt(dot(x0, x0))
    indx = r>R
    E = np.zeros_like(r)
    E[indx] = K/2 * (r[indx] - R)**2
    return E

def f(x0):
    r = sqrt(dot(x0, x0))
    indx = r>R
    f = np.zeros_like(x0)
    f[:, indx] = -K*(r[indx] - R) * x0[:, indx] / r[indx]
    return f


def verletStep(x):
    f1 = f(x[0])
    x0 = x[0] + x[1]*dt + 1/2*f1*dt**2
    x1 = x[1] + (f1 + f(x0)) / 2. * dt 
    return np.array([x0, x1])

def E(x):
    return V(x[:, 0]) + 1/2*dot(x[:, 1], x[:, 1])

def ETotal(x):
    return np.einsum("ij -> i", E(x))

# Starting poistionspritn
# Places N particles randomly, with random speeds
# x[i, j, k] is i'th derivative of coordinate j of particle k
x0 = np.empty((2, 2, N))
x0[0] = np.random.uniform(-R, R, (2, N))
angles = np.random.uniform(-pi, pi, N)
x0[1] = np.array([cos(angles), sin(angles)])

x = np.zeros((timeSteps, *np.shape(x0)))
x[0] = x0

t = time()
for i in range(timeSteps-1):
    x[i+1] = verletStep(x[i])
print(time() - t)

theta = np.linspace(0, 2*pi, 100)
X, Y = R*cos(theta), R*sin(theta)
plt.plot(X, Y)
for i in range(N):
    plt.plot(x[0, 0, 0, i], x[0, 0, 1, i], "rx")
    plt.plot(x[:, 0, 0, i], x[:, 0, 1, i], color = cm.viridis(i/N))
plt.show()

energy =  ETotal(x)
relEnergy = (energy - energy[0]) / energy[0]
plt.plot(np.linspace(0, dt*timeSteps, timeSteps), relEnergy)
plt.show()
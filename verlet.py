import numpy as np
from numpy import cos, sin, pi, sqrt
from time import time
# Neew to know phys.param. for placing particles
from physics import a, R


# Places N particles randomly, but not ontop of echother
def distribute(N):
    print("placing particles...")
    x0 = np.empty((2, N))
    x0[:, 0] = np.random.uniform(-R/sqrt(2), R/sqrt(2), 2)

    k = 1
    while k<N:
        xt0k = np.random.uniform(-R/sqrt(2), R/sqrt(2), 2)
        for j in range(k):
            r = x0[:, j] - xt0k
            if r@r>a**2:
                pass
            else:
                break
            if j == k-1:
                x0[:, k] = xt0k
                k += 1
    return x0


# Different vel distributions for the particles
# Distirubutes velocity uniformally
def evenVel(N):
    theta = np.random.uniform(-pi, pi, N)
    return np.array([cos(theta), sin(theta)])

# Gives all the velocity to one particle
def allVelToOne(N):
    vel = np.zeros((2, N))
    theta, = np.random.uniform(-pi, pi, 1)
    vel[:, 0] = np.array([cos(theta), sin(theta)]) * sqrt(N)
    return vel

def verletStep(x, F, dt):
    f1 = F(x[0])
    x0 = x[0] + x[1]*dt + 1/2*f1*dt**2
    x1 = x[1] + (f1 + F(x0)) / 2 * dt 
    return np.array([x0, x1])

# Simulate the trajectories of N particles subject to forces
# given by F
def timeEvolve(N, F, timeSteps, dt, velDist = evenVel):
    xt = np.zeros((timeSteps, 2, 2, N))
    xt[0, 0] = distribute(N)
    xt[0, 1] = velDist(N)

    t = time()
    print("Integrating steps...")
    for i in range(timeSteps-1):
        xt[i+1] = verletStep(xt[i], F, dt)
    t = time() - t

    print("Integreated {} steps of {} particels in {:f} sec.".format(timeSteps, N, t))
    return xt
import numpy as np
from numpy import cos, sin, pi
from time import time
# Neew to know phys.param. for placing particles
from physics import a, R

# Places N particles randomly, but not ontop of echother, with speeds
# x[i, j, k] is i'th derivative of coordinate j of particle k
def getEmpty(N, timeSteps):
    print("placing particles...")
    xt0 = np.empty((2, 2, N))
    xt0[0, :, 0] = np.random.uniform(-R, R, 2)

    k = 1
    while k<N:
        xt0k = np.random.uniform(-R, R, 2)
        for j in range(k):
            r = xt0[0, :, j] - xt0k
            if r@r>a**2:
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
    return xt

def verletStep(x, F, dt):
    f1 = F(x[0])
    x0 = x[0] + x[1]*dt + 1/2*f1*dt**2
    x1 = x[1] + (f1 + F(x0)) / 2 * dt 
    return np.array([x0, x1])

# Simulate the trajectories of N particles subject to forces
# given by F, 
def timeEvolve(N, F, timeSteps, dt):
    xt = getEmpty(N, timeSteps)
    t = time()
    print("Integrating steps...")
    for i in range(timeSteps-1):
        xt[i+1] = verletStep(xt[i], F, dt)
    t = time() - t
    print("Integreated {} steps of {} particels in {:f} sec.".format(timeSteps, N, t))
    return xt
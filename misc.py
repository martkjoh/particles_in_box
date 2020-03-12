import numpy as np
from math import floor
import os
from physics import sumE

def savex(xt, T, name, E = sumE):
    path = "data/" + name + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    steps = len(xt)
    stepSize = 1
    # around 1000 points is enough
    if steps>1000:
        stepSize = floor(steps/1000)
    xt = xt[::stepSize]
    steps = len(xt)
    N = len(xt[0, 0, 0])

    energy = np.empty((steps, N))
    for t in range(steps):
        energy[t] = E(xt[t])
    
    xt = np.reshape(xt, (steps, 4*N))

    header = "Time simulated, number of steps and number of particles"

    np.savetxt(path + "particles" + ".csv", xt)
    np.savetxt(path + "energy" + ".csv", energy)
    np.savetxt(
        path + "parametres.csv", 
        [T, steps, N], 
        delimiter=",", 
        header = header)
        

def loadx(name):
    path = path = "data/" + name + "/" 
    xt = np.loadtxt(path + "particles" + ".csv")
    E = np.loadtxt(path + "energy" + ".csv")
    T, steps, N = np.loadtxt(
        path + "parametres.csv", 
        skiprows = 1, 
        unpack = True)

    dt = T / steps
    N = int(N)
    steps = int(steps)
    
    E = np.reshape(E, (steps, N))
    print("Loaded {} particles with {} steps".format(N, steps))
    return np.reshape(xt, (steps, 2, 2, N)), E, T, dt, N

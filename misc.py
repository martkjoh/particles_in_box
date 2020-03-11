import numpy as np
from math import floor
import os

def savex(xt, T, dt, folder, name):
    steps = len(xt)
    stepSize = floor(steps/1000)
    # around 1000 points is enough
    xt = xt[::stepSize]
    steps = len(xt)
    N = len(xt[0, 0, 0])
    xt = np.reshape(xt, (steps, 4*N))

    path = "data_" + folder + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    header = "Time simulated, number of steps and number of particles"
    
    np.savetxt(
        path + "parametres.csv", 
        [T, steps, N], 
        delimiter=",", 
        header = header)
        
    np.savetxt(path + name + ".csv", xt)

def loadx(folder, name):
    path = path = "data_" + folder + "/" 
    if not os.path.exists("data_"+folder+"/"+name+".csv"):
        print("Can't load file")
        return

    T, steps, N = np.loadtxt(path + "parametres.csv", unpack = True)
    xt = np.loadtxt(path + name + ".csv")
    dt = T / steps
    print("Loaded {} particles with {} steps".format(N, steps))

    return np.reshape(xt, (t, 2, 2, N)), T, dt

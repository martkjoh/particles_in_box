import numpy as np
import os

def savex(xt, T, dt, folder, name):
    if not os.path.exists("data_" + folder):
        os.makedirs("data_" + folder + "/")
    t = len(xt)
    N = len(xt[0, 0, 0])
    xt = np.reshape(xt, (t, 4*N))
    np.savetxt("data_"+folder+"/"+"parametres.csv", [T, dt], delimiter=",")
    np.savetxt("data_"+folder+"/"+name+".csv", xt)

def loadx(folder, name):
    if not os.path.exists("data_"+folder+"/"+name+".csv"):
        print("Can't load")
    T, dt = np.loadtxt("data_"+folder+"/"+"parametres.csv", unpack = True)
    xt = np.loadtxt("data_"+folder+"/"+name+".csv")
    t = len(xt)
    N = int(len(xt[0])/4)

    return np.reshape(xt, (t, 2, 2, N)), T, dt

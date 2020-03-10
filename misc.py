import numpy as np
import os

def savex(xt, folder, name):
    if not os.path.exists("data_" + folder):
        os.makedirs("data_" + folder + "/")
    t = len(xt)
    N = len(xt[0, 0, 0])
    xt = np.reshape(xt, (t, 4*N))
    np.savetxt("data_" + folder + "/" + name + ".csv", xt)

def loadx(folder, name):
    if not os.path.exists("data_"+folder+"/"+name+".csv"):
        print("Can't load")
    xt = np.loadtxt("data_" + folder + "/" + name + ".csv")
    t = len(xt)
    N = len(xt[0])

    return np.reshape(xt, (t, 2, 2, int(N/4)))

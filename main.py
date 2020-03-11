from physics import sumF, F_w, temp
from verlet import timeEvolve, allVelToOne, evenVel
from misc import savex, loadx
from plot import *
from matplotlib import pyplot as plt
from time import sleep

def runSimulation(N, T, dt, name, velDist=evenVel):
    timeSteps = int(T/dt)
    xt = timeEvolve(N, sumF, timeSteps, dt, velDist=velDist)
    savex(xt, T, name)

def task1a():
    dts = [0.1, 0.05, 0.01]


    for dt in dts:
        name = "one_particle_{}".format(dt).replace(".", "")

        # runSimulation(1, 100, dt, name)

        xt, E, T, dt_new, N = loadx(name)

        fig, ax = plt.subplots(1, 2, figsize = (14, 7))
        fig.suptitle("$\Delta t = {0:1.3f}$".format(dt))
        plotCircle(ax[0])
        plotParticlePath(xt, ax[0], 0, color = cm.viridis(0.2))
        plotRelTotE(E, T, dt, ax[1])

        plt.tight_layout()
        plt.savefig("figs/"+name+".png")


task1a()

# runSimulation(10, 10, 0.01, "test")
# xt, T, dt, N = loadx("test")

# # Regular plot, all
# for t in range(10):
#     fig, ax = plt.subplots()
#     plotCircle(ax)
#     for i in range(N):
#         plotParticlePath(xt[100*t:100*(t+1)], ax, i, color=cm.viridis(i/N))
#     plt.show()

# velDistirbute
# xAv = np.array([np.einsum("tijk -> tij", xt)]) / N
# fig, ax = plt.subplots()
# plotVelDistrib(xt, 0, ax)

# plt.show()

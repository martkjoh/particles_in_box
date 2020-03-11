from physics import sumF, F_w, temp
from verlet import timeEvolve, allVelToOne
from misc import savex, loadx
from plot import *
from matplotlib import pyplot as plt
from time import sleep

def runSimulation(N, T, dt):
    timeSteps = int(T/dt)
    xt = timeEvolve(N, sumF, timeSteps, dt, allVelToOne)
    savex(xt, T, dt, "a", "b")

# runSimulation(50, 100, 0.01)

xt, T, dt = loadx("allVelToOne", "particles")
N = len(xt[0, 0, 0])

# # Regular plot, all
# for t in range(10):
#     fig, ax = plt.subplots()
#     plotCircle(ax)
#     for i in range(N):
#         plotParticlePath(xt[100*t:100*(t+1)], ax, i, color=cm.viridis(i/N))
#     plt.show()

# # velDistirbute
# fig, ax = plt.subplots()
# plotVelDistrib(xt, 4, ax)
# plt.show()

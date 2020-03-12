from physics import sumF, F_w, temp, sumE, E_k
from verlet import timeEvolve, allVelToOne, evenVel
from misc import savex, loadx
from plot import *
from matplotlib import pyplot as plt
from time import sleep

def runSimulation(N, T, dt, name, velDist=evenVel, E = sumE):
    timeSteps = int(T/dt)
    xt = timeEvolve(N, sumF, timeSteps, dt, velDist=velDist)
    savex(xt, T, name, E = sumE)

def task1():
    dts = [0.1, 0.05, 0.01]

    for dt in dts:
        name = "one_particle_{}".format(dt).replace(".", "")

        # runSimulation(1, 100, dt, name)

        xt, E, T, dt_new, N = loadx(name)

        fig, ax = plt.subplots(1, 2, figsize = (14, 7))
        fig.suptitle("$\Delta t = {0:1.3f}$".format(dt))
        plotCircle(ax[0])
        plotParticlePath(xt, ax[0], 0, color = cm.viridis(0.2), title = Position)
        plotRelTotE(E, T, dt, ax[1])

        plt.tight_layout()
        plt.savefig("figs/"+name+".png")

def task2():
    Ns = [2, 5, 10]
    T = 50
    fig, ax = plt.subplots(3, 2, figsize = (14, 21))
    for i in range(len(Ns)):
        name = "{}_particles".format(Ns[i])
        
        # runSimulation(Ns[i], T, 0.01, name)

        xt, E, T, dt, N = loadx(name)

        plotCircle(ax[i, 0])
        for k in range(Ns[i]):
            plotParticlePath(
                xt, ax[i, 0], k, 
                title = "${}$ particles".format(N), 
                color = cm.viridis(k/N))
            plotRelTotE(E, T, dt, ax[i, 1])
        
    plt.tight_layout()
    plt.savefig("figs/several_particles.png")

def scatter():
    fig, ax = plt.subplots(figsize = (10, 10))
    N = 50
    T = 1000
    name = "{}_particles_time_{}".format(N, T)

    # runSimulation(N, T, 0.01, name)

    xt, E, T, dt, N = loadx(name)

    title = "${}$ particles, T = ${}$".format(N, T)
    plotCircle(ax)
    for k in range(N):
        plotScatter(xt, k, ax, title = title, color = cm.viridis(k/N))
        
    plt.savefig("figs/scatter.png")

def task2c():
    T = 100
    N = 50
    dt = 0.01
    name = "all_vel_to_one"

    # runSimulation(N, T, dt, name, velDist=allVelToOne, E = E_k)

    xt, E, T, dt, N = loadx(name)

    fig, ax = plt.subplots(2, figsize = (7, 7))
    plotEDist(E, T, ax)

    plt.tight_layout()
    plt.savefig("figs/" + name)

task2c()


# velDistirbute
# xAv = np.array([np.einsum("tijk -> tij", xt)]) / N
# fig, ax = plt.subplots()
# plotVelDistrib(xt, 0, ax)

# plt.show()

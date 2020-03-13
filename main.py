from physics import sumF, F_w, getTemp, sumE, E_k
from verlet import timeEvolve, allVelToOne, evenVel
from misc import savex, loadx
from plot import *
from matplotlib import pyplot as plt

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
        plotParticlePath(xt, ax[0], 0, color = cm.viridis(0.2), title = "Position")
        plotRelTotE(E, T, dt, ax[1])

        plt.tight_layout()
        plt.savefig("figs/"+name+".png")

def task2a():
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

def task2b():
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

def task2d():
    # Data from scatter
    name = "50_particles_time_1000"

    fig, ax = plt.subplots( figsize = (12, 10))

    xt, E, T, dt, N = loadx(name)

    plotVelDistrib(xt, 0, ax)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/vel_dist.png")

def task2e(R, N):
    # Data from scatter

    # T = 200
    # dt = 0.01

    name = "{}_particles_R_{}".format(N, R)

    # runSimulation(N, T, dt, name)

    xt, _E, T, dt, N = loadx(name)
    _fig, ax = plt.subplots(figsize = (7, 7))
    
    title = "$N = {}$, $R = {}$".format(N, R) 
    plotPressure(xt, T, ax, title = title)

    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/pressure_" + name + ".png")

# R needs to be to changed in physics.py

# task2e(2, 10)
# task2e(2, 12)
# task2e(4, 20)
# task2e(4, 25)

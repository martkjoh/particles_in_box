import numpy as np
from numpy import pi, sin, cos, sqrt
from matplotlib import pyplot as plt
from matplotlib import cm
from physics import R, boltzDist, pressure, getTemp


font = {'family' : 'serif', 
        'weight' : 'normal', 
        'size'   : 18}
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rc("lines", lw=2)
plt.rc('font', **font)

def plotCircle(ax):
    theta = np.linspace(0, 2*pi, 100)
    ax.plot(R*cos(theta), R*sin(theta))


def plotParticlePath(xt, ax, k, color = "b", title = ""):
    ax.plot(xt[0, 0, 0, k], xt[0, 0, 1, k], "rx")
    ax.plot(xt[:, 0, 0, k], xt[:, 0, 1, k], color = color)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

def plotRelTotE(E, T, dt, ax):
    timeSteps = len(E)
    t = np.linspace(0, T, timeSteps)
    totEnergy = np.einsum("tk -> t", E)
    relEnergy = (totEnergy - totEnergy[0]) / totEnergy[0]
    ax.plot(t, relEnergy)
    ax.set_ylim(-0.1, 0.1)
    ax.set_title("Relative change in total energy")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\Delta E / E(t)$")

def plotVelDistrib(xt, k, ax):
    vx = xt[:, 1, 0, k]
    ax.hist(vx, bins = 15, density = True)
    vxCont = np.linspace(np.min(vx), np.max(vx), 100)
    f = boltzDist(xt)(vxCont)
    ax.plot(vxCont, f, label = "Boltzmann dist.")
    ax.set_xlabel("$v_x$")
    ax.set_ylabel("Relative frequency")
    

def plotScatter(xt, k, ax, color = "b", title = ""):
    ax.scatter(xt[:, 0, 0, k], xt[:, 0, 1, k], color = color, alpha = 0.1)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

def plotEDist(E, T, ax):
    N = len(E[0])
 
    steps = len(E)
    t = np.linspace(0, T, steps)
    EAv = np.einsum("tk -> k", E[900:]) / N
    ax[0].hist(E[0], bins = 20)
    ax[0].set_title("$T = 0$")
    ax[0].set_xlabel("$E$")
    ax[0].set_ylabel("N")
    ax[1].hist(E[-1], bins = 20)
    ax[1].set_title("Average, $T \in [90, 100]$")
    ax[1].set_xlabel("$E$")
    ax[1].set_ylabel("N")

    plt.tight_layout()

# TODO: kAv should always be one, but isn't. Fix this
def plotPressure(xt, T, ax):
    steps = len(xt)
    N = len(xt[0, 0, 0])

    P = np.empty(steps)
    temp = 0
    for t in range(steps):
        P[t] = pressure(xt[t, 0])
        temp += getTemp(xt[t])
    
    temp = temp/steps

    t = np.linspace(0, T, steps)

    k = P*(pi*R**2)/temp/N
    kAv = np.sum(k) / steps
    ax.plot(t, k, label = "$P / T A$")
    ax.plot(t, kAv*np.ones(steps), label = "Average value = {:f}".format(kAv))

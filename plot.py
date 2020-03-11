import numpy as np
from numpy import pi, sin, cos, sqrt
from matplotlib import pyplot as plt
from matplotlib import cm
from physics import sumE, getEnergy, R, boltzDist


font = {'family' : 'serif', 
        'weight' : 'normal', 
        'size'   : 18}
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rc("lines", lw=2)
plt.rc('font', **font)

def plotCircle(ax):
    theta = np.linspace(0, 2*pi, 100)
    ax.plot(R*cos(theta), R*sin(theta))


def plotParticlePath(xt, ax, k, color = "b"):
    ax.plot(xt[0, 0, 0, k], xt[0, 0, 1, k], "rx")
    ax.plot(xt[:, 0, 0, k], xt[:, 0, 1, k], color = color)
    ax.set_title("Postion")
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
    vx, _ = xt[:, 1, :, k].T
    ax.hist(vx, bins = 50, density = True)
    vxCont = np.linspace(np.min(vx), np.max(vx), 100)
    f = boltzDist(xt)(vxCont)
    ax.plot(vxCont, f)
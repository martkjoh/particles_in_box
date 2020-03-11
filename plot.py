import numpy as np
from numpy import pi, sin, cos, sqrt
from matplotlib import pyplot as plt
from matplotlib import cm
from physics import sumE, getEnergy, R, boltzDist

def plotCircle(ax):
    theta = np.linspace(0, 2*pi, 100)
    ax.plot(R*cos(theta), R*sin(theta))


def plotParticlePath(xt, ax, k, color = "b"):
    ax.plot(xt[0, 0, 0, k], xt[0, 0, 1, k], "rx")
    ax.plot(xt[:, 0, 0, k], xt[:, 0, 1, k], color = color)

def plotRelTotE(xt, T, dt, ax):
    timeSteps = int(T/dt)
    t = np.linspace(0, T, timeSteps)
    energy = getEnergy(xt, sumE)
    totEnergy = np.einsum("tk -> t", energy)
    relEnergy = (totEnergy - totEnergy[0]) / totEnergy[0]
    ax.plot(t, relEnergy)
    ax.set_ylim(-0.1, 0.1)

def plotVelDistrib(xt, k, ax):
    vx, _ = xt[:, 1, :, k].T
    ax.hist(vx, bins = 50, density = True)
    vxCont = np.linspace(np.min(vx), np.max(vx), 100)
    f = boltzDist(xt)(vxCont)
    ax.plot(vxCont, f)
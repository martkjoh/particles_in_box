import numpy as np
from numpy import pi, sin, cos
from misc import loadx
from matplotlib import pyplot as plt
from matplotlib import cm
from physics import sumE, getEnergy, R

def plotCircle(ax):
    theta = np.linspace(0, 2*pi, 100)
    ax.plot(R*cos(theta), R*sin(theta))


def plotParticlePath(xt, ax, i, color = "b"):
    ax.plot(xt[0, 0, 0, i], xt[0, 0, 1, i], "rx")
    ax.plot(xt[:, 0, 0, i], xt[:, 0, 1, i], color = color)

def plotRelTotE(xt, T, dt, ax):
    timeSteps = int(T/dt)
    t = np.linspace(0, T, timeSteps)
    energy = getEnergy(xt, sumE)
    totEnergy = np.einsum("tk -> t", energy)
    relEnergy = (totEnergy - totEnergy[0]) / totEnergy[0]
    ax.plot(t, relEnergy)
    ax.set_ylim(-0.1, 0.1)

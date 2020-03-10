import numpy as np
from numpy import pi, sin, cos
from matplotlib import pyplot as plt
from matplotlib import cm
from physics import sumF, sumE, getEnergy, R, dot
from verlet import timeEvolve

N = 50
dt = 0.01
T = 10
timeSteps = int(T/dt)
xt = timeEvolve(N, sumF, timeSteps , dt)

theta = np.linspace(0, 2*pi, 100)
plt.plot(R*cos(theta), R*sin(theta))
for i in range(N):
    plt.plot(xt[0, 0, 0, i], xt[0, 0, 1, i], "rx")
    plt.plot(xt[:, 0, 0, i], xt[:, 0, 1, i], color = cm.viridis(i/N))

plt.show()
energy = getEnergy(xt, sumE)
totEnergy = np.einsum("tk -> t", energy)
relEnergy = (totEnergy - totEnergy[0]) / totEnergy[0]
plt.plot(np.linspace(0, dt*timeSteps, timeSteps), relEnergy)
plt.ylim(-0.1, 0.1)
plt.show()

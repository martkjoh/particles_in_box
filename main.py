import numpy as np
from numpy import pi, sin, cos
from matplotlib import pyplot as plt
from matplotlib import cm
from physics import sumF, sumE, getEnergy, R, dot
from verlet import *
from misc import savex

N = 5
dt = 0.01
T = 10
timeSteps = int(T/dt)
xt = timeEvolve(N, sumF, timeSteps , dt)
savex(xt, "test", "test")

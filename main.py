from physics import sumF, F_w
from verlet import timeEvolve
from misc import savex


def runSimulation():
    N = 10
    dt = 0.01
    T = 10
    timeSteps = int(T/dt)
    xt = timeEvolve(N, sumF, timeSteps , dt)
    savex(xt, T, dt, "test", "test")


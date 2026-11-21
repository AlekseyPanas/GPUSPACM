import pygame
from matplotlib import pyplot as plt


class Sim1D:
    def __init__(self, gravity: float, mass: float):
        self._gravity = gravity
        self._mass = mass
        self._floor = 0
        self._pos = [1000.0]
        self._vel = [0.0]
        self._t = [0.0]
        self._total_energy = [self.get_total_energy()]

    def step(self, delta_t: float):
        self._pos.append( self._pos[-1] + (0.5 * self._gravity * (delta_t ** 2)) + (self._vel[-1] * delta_t) )
        self._vel.append( self._vel[-1] + self._gravity * delta_t )
        self._t.append(self._t[-1] + delta_t)
        self._total_energy.append(self.get_total_energy())

    def get_total_energy(self):
        def potential():
            return self._mass * -self._gravity * self._pos[-1]

        def kinetic():
            return 0.5 * self._mass * (self._vel[-1] ** 2)

        return potential() + kinetic()

    def plot_energy(self):
        plt.figure(figsize=(8, 8))
        plt.title("Energy")
        plt.plot(self._total_energy)
        plt.show()

    def plot_pos(self):
        plt.figure(figsize=(8, 8))
        plt.title("Position")
        plt.plot(self._pos)
        plt.show()


sim = Sim1D(-9.81, 10)
for _ in range(10):
    sim.step(1)
sim.plot_energy()
sim.plot_pos()

print(sim._total_energy)

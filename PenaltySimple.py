import pygame
import math
from matplotlib import pyplot as plt
from dataclasses import dataclass
from typing import Optional


@dataclass
class Snapshot:
    p: float
    v: float
    t: float
    energy: Optional[float]


def snapSetEnergy(snap: Snapshot, M: float, G: float, floor_p: float):
    snap.energy = (0.5 * M * snap.v ** 2) + (M * G * (snap.p - floor_p))


def takeStep(cur_snap: Snapshot, dt: float, G: float, floor_p: float, M: float, penalty_thickness: float) -> Snapshot:
    """
    Symplectic:
    x_t+1 = x_t + v_t * dt
    v_t+1 = v_t + F(x_t+1)/M * dt
    """
    def get_penalty(pos: float):
        dist_from_penalty = pos - (floor_p + penalty_thickness)
        if dist_from_penalty <= 0:
            return 6 ** abs(dist_from_penalty)
        else:
            return 0

    p_t1 = cur_snap.p + cur_snap.v * dt

    Fg = M * G
    Fpenalty = get_penalty(p_t1)
    Fnet = Fg + Fpenalty

    v_t1 = cur_snap.v + (Fnet / M) * dt

    snap = Snapshot(p_t1, v_t1, cur_snap.t + dt, None)
    snapSetEnergy(snap, M, G, floor_p)
    return snap


def runSim(firstSnapshot: Snapshot, G: float, M: float, floor_p: float,
           timestep: float, penalty_thickness: float) -> list[Snapshot]:
    pygame.init()

    snapSetEnergy(firstSnapshot, M, G, floor_p)
    snapshots = [firstSnapshot]

    clock = pygame.time.Clock()
    mult = 2
    screen = pygame.display.set_mode((300 * mult, 300 * mult), pygame.DOUBLEBUF)

    timeStepAlways = False
    running = True
    while running:
        screen.fill((255, 255, 255))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_c:
                    timeStepAlways = not timeStepAlways
                if not timeStepAlways and ev.key == pygame.K_SPACE:
                    snapshots.append(takeStep(snapshots[-1], timestep, G, floor_p, M, penalty_thickness))

        if timeStepAlways:
            snapshots.append(takeStep(snapshots[-1], timestep, G, floor_p, M, penalty_thickness))

        pygame.draw.circle(screen, (255, 0, 0), (150 * mult, (250 - (snapshots[-1].p * 10)) * mult), 4 * mult)
        pygame.draw.line(screen, (0, 0, 255), (0, 250 * mult), (300 * mult, 250 * mult))

        pygame.display.update()

        clock.tick(100)

    return snapshots


if __name__ == "__main__":
    initSnap = Snapshot(
        p=10,  # m
        v=0,   # m / s
        t=0,
        energy=None
    )

    snapshots = runSim(
        initSnap,
        G=-7,
        M=1,
        floor_p=0,
        timestep=0.01,
        penalty_thickness=3
    )

    plt.plot([s.p for s in snapshots])
    plt.show()
    plt.plot([s.v for s in snapshots])
    plt.show()
    plt.plot([s.energy for s in snapshots])
    plt.show()

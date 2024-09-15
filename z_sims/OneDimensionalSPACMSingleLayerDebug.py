from __future__ import  annotations

import heapq

from core import *
from z_logging.loggers import Logger


class InfinitePenaltyEvent(Event):
    """Infinitely large penalty layer active below some threshold"""

    def get_identifier(self) -> int:
        return 2

    def __init__(self, thresh: float, timestep: float, stiffness: float):
        super().__init__(timestep)
        self.thresh = thresh
        self.stiffness = stiffness

    def get_gap(self, q: float):
        return q - self.thresh

    def get_force(self, target: Particle):
        gap = target.get_latest_position() - self.thresh
        if gap <= 0:
            return self.stiffness * abs(gap)
        else:
            return 0

    def get_stiffness(self):
        return self.stiffness


class SPACM1DSimInfiniteLayer(Sim):
    def __init__(self, rollback_window_size: float, accel_gravity: float, timestep_gravity: float,
                 particle_positions: list[float], particle_velocities: list[float],
                 particle_masses: list[float], collision_stiffness: float, penalty_layer_thickness: float,
                 penalty_timestep: float, logger: Logger, compute_energy: bool = True,
                 gravity_energy_base: float = -10):
        assert len(particle_positions) == len(particle_velocities)

        self.logger = logger

        self.past_events: list[tuple[float, Event]] = []  # History for logging

        self.R = rollback_window_size
        self.penalty_timestep = penalty_timestep

        self.eventQ: list[tuple[float, Event]] = []

        self.accel_gravity = accel_gravity

        def stiffness_lth(l: int):
            return collision_stiffness * (l ** 3)  # ACM paper section 4

        def timestep_lth(l: int, fudge=1e-4):
            return self.penalty_timestep / l / math.sqrt(l + fudge)

        def thickness_lth(l: int):
            return 10 * (l ** (-0.25))

        self.inf_penalty_events = [
            InfinitePenaltyEvent(0, penalty_timestep, stiffness_lth(1)),
            InfinitePenaltyEvent(0 - (thickness_lth(1) - thickness_lth(2)), timestep_lth(2), stiffness_lth(2)),
            InfinitePenaltyEvent(0 - (thickness_lth(1) - thickness_lth(3)), timestep_lth(3), stiffness_lth(3))
        ]

        # Schedule first gravity event
        heapq.heappush(self.eventQ, (timestep_gravity, GravityEvent(timestep_gravity, accel_gravity)))
        # Schedule infinite penalty
        for pen in self.inf_penalty_events:
            heapq.heappush(self.eventQ, (penalty_timestep, pen))

        self.start_of_window = 0
        self.end_of_window = self.R
        self.eventQ_at_start = [(t, e) for t, e in self.eventQ]  # Copies event queue

        self.particles: list[Particle] = [Particle(collision_stiffness, penalty_layer_thickness,
                                                   particle_positions[i], particle_velocities[i],
                                                   0, particle_masses[i]) for i in range(len(particle_positions))]

        self.do_compute_energy = compute_energy  # Should energy be tracked at each snapshot
        self.gravity_base = gravity_energy_base  # Vertical position of "ground" relative to which to compute gravitational energy

    def run_sim(self):
        while True:
            te, e = heapq.heappop(self.eventQ)  # time, event

            for p in self.particles:
                p0 = p.current_snapshots[-1]

                x1 = p0.x + ((te - p0.t) * p0.v)  # integrate position up to this event
                t1 = te  # update to new time

                p.current_snapshots.append(Snapshot(x1, -1, t1, None, False, None, None, None,
                                                    e.get_identifier()))  # Add new snapshot with placeholder v and has_velocity_changed

            for p in self.particles:
                p0 = p.current_snapshots[-2]
                pc = p.current_snapshots[-1]

                v1_5 = p0.v + (e.h / 2) * (e.get_force(p) / p.mass)  # record energy at halfway vel update
                v1 = p0.v + e.h * (e.get_force(p) / p.mass)  # integrate force with respect to force's timestep

                # Update placeholder v and has_velocity_changed values
                pc.has_velocity_changed = p0.v != v1
                pc.v = v1

                # Update energy
                if self.do_compute_energy:
                    Eg = p.mass * abs(self.accel_gravity) * (pc.x - self.gravity_base)  # E_gravity = mgh
                    Ek = 0.5 * p.mass * (v1_5 ** 2)  # E_kinetic = 1/2 mv^2
                    Ep = 0
                    for pen in self.inf_penalty_events:
                        gap = pen.get_gap(p.get_latest_position())
                        Ep += 0 if gap > 0 else 0.5 * pen.get_stiffness() * (gap ** 2)
                    Etotal = Eg + Ek + Ep
                    pc.energy = Etotal

            # Event-granular log: latest snapshot of all particles
            self.logger.record_snapshots([p.current_snapshots[-1] for p in self.particles])

            heapq.heappush(self.eventQ, (te + e.h, e))  # Schedule next force event

            self.past_events.append((te, e))  # Record processed event

            yield YieldType.EVENT_FINISHED, self

    def output_log_data(self):
        self.logger.output_data()

    def get_collideables(self) -> list[Collidable]:
        return self.particles

    def get_particles(self) -> list[Particle]:
        return self.particles

    def get_walls(self) -> list[Collidable]:
        return []

    def get_window_size(self) -> float:
        return self.R

    def get_eventQ(self) -> list[tuple[float, Event]]:
        return self.eventQ

    def get_past_events(self) -> list[tuple[float, Event]]:
        return self.past_events

    def get_logger(self) -> Logger:
        return self.logger
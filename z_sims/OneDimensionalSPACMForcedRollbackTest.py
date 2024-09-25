from __future__ import  annotations

import heapq

from z_sims.core.sim import Sim
from z_sims.core.events import Event, GravityEvent, PenaltyEvent
from z_sims.core.objects import Collidable, Particle
from z_sims.core.data import YieldType, Snapshot
from z_sims.core.util import get_next_time
from z_logging.loggers import Logger


class SPACM1DSimForcedRollback(Sim):
    def __init__(self, rollback_window_size: float, accel_gravity: float, timestep_gravity: float,
                 particle_positions: list[float], particle_velocities: list[float], wall_positions: list[float],
                 particle_masses: list[float], collision_stiffness: float, penalty_layer_thickness: float,
                 penalty_timestep: float, logger: Logger, compute_energy: bool = True,
                 gravity_energy_base: float = -10):
        assert len(particle_positions) == len(particle_velocities)

        self.logger = logger
        self.logger.record_config(locals())

        self.past_events: list[tuple[float, Event]] = []  # History for logging

        self.R = rollback_window_size
        self.penalty_timestep = penalty_timestep

        self.eventQ: list[tuple[float, Event]] = []

        self.accel_gravity = accel_gravity

        # Schedule first gravity event
        heapq.heappush(self.eventQ, (timestep_gravity, GravityEvent(timestep_gravity, accel_gravity)))

        self.start_of_window = 0
        self.end_of_window = self.R
        self.eventQ_at_start = [(t, e) for t, e in self.eventQ]  # Copies event queue

        self.particles: list[Particle] = [Particle(collision_stiffness, penalty_layer_thickness,
                                                   particle_positions[i], particle_velocities[i],
                                                   0, particle_masses[i]) for i in range(len(particle_positions))]
        self.walls: list[Collidable] = [Collidable(collision_stiffness,
                                                          penalty_layer_thickness,
                                                          p) for p in wall_positions]
        self.collideables: list[Collidable] = self.walls + self.particles

        self.do_compute_energy = compute_energy  # Should energy be tracked at each snapshot
        self.gravity_base = gravity_energy_base  # Vertical position of "ground" relative to which to compute gravitational energy

    def run_sim(self):
        while True:
            while self.eventQ[0][0] <= self.end_of_window:
                te, e = heapq.heappop(self.eventQ)  # time, event

                for p in self.particles:
                    p0 = p.current_snapshots[-1]

                    x1 = p0.x + ((te - p0.t) * p0.v)  # integrate position up to this event
                    t1 = te  # update to new time

                    p.current_snapshots.append(Snapshot(x1, -1, t1, None, False, None, None, None, e.get_identifier()))  # Add new snapshot with placeholder v and has_velocity_changed

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
                        for c in self.collideables:
                            if c is not p:
                                for pen in c.active_penalties:
                                    gap = pen.get_gap(p.get_latest_position())
                                    if gap <= 0:
                                        Ep += 0.5 * c.get_lth_stiffness(pen.layer) * (gap ** 2)
                        Etotal = Eg + Ek + Ep
                        pc.energy = Etotal
                        pc.kinetic_energy = Ek
                        pc.potential_energy = Eg
                        pc.penalty_energy = Ep

                # Event-granular log: latest snapshot of all particles
                self.logger.record_snapshots([p.current_snapshots[-1] for p in self.particles])

                heapq.heappush(self.eventQ, (te + e.h, e))  # Schedule next force event

                self.past_events.append((te, e))  # Record processed event

                yield YieldType.EVENT_FINISHED, self

            # Linearly update particle positions and timestamps to the end of the rollback window
            for p in self.particles:
                p0 = p.current_snapshots[-1]
                if p0.t < self.end_of_window:
                    x1 = p0.x + ((self.end_of_window - p0.t) * p0.v)
                    t1 = self.end_of_window
                    p.current_snapshots.append(Snapshot(x1, p0.v, t1, p0.energy, False,
                                                        p0.kinetic_energy, p0.potential_energy, p0.penalty_energy, None))

            # window-granular log for particles
            self.logger.record_window_snapshots([p.current_snapshots[-1] for p in self.particles])

            # Initiate rollback
            if True:
                # Rollback
                for p in self.particles:
                    p.current_snapshots = [p.current_snapshots[0]]  # Reset snapshots
                self.eventQ = [(t, e) for t, e in self.eventQ_at_start]  # Restore initial event queue

                # Update saved starting event queue
                self.eventQ_at_start = [(t, e) for t, e in self.eventQ]

                # Notify logger that rollback has occurred
                self.logger.rollback()

                yield YieldType.TIME_WINDOW_FINISHED_ROLLBACK, self

    def output_log_data(self):
        self.logger.output_data()

    def get_collideables(self) -> list[Collidable]:
        return self.collideables

    def get_particles(self) -> list[Particle]:
        return self.particles

    def get_walls(self) -> list[Collidable]:
        return self.walls

    def get_window_size(self) -> float:
        return self.R

    def get_eventQ(self) -> list[tuple[float, Event]]:
        return self.eventQ

    def get_past_events(self) -> list[tuple[float, Event]]:
        return self.past_events

    def get_logger(self) -> Logger:
        return self.logger

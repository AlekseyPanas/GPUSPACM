from __future__ import  annotations

import heapq

from z_sims.core.sim import Sim
from z_sims.core.events import Event, GravityEvent, PenaltyEvent
from z_sims.core.objects import Collidable, Particle
from z_sims.core.data import YieldType, Snapshot
from z_sims.core.util import get_next_time, get_next_index
from z_logging.loggers import Logger


class SPACM1DSim(Sim):
    def __init__(self, rollback_window_size: float, accel_gravity: float, timestep_gravity: float,
                 particle_positions: list[float], particle_velocities: list[float], wall_positions: list[float],
                 particle_masses: list[float], collision_stiffness: float, penalty_layer_thickness: float,
                 penalty_timestep: float, logger: Logger, compute_energy: bool = True,
                 gravity_energy_base: float = -10):
        assert len(particle_positions) == len(particle_velocities)

        self.logger = logger
        self.logger.record_config(locals())

        self.past_events: list[tuple[float, int, Event]] = []  # History for logging

        self.R = rollback_window_size
        self.penalty_timestep = penalty_timestep

        self.eventQ: list[tuple[float, int, Event]] = []  # Time, timestep index (i.e time = timestep_index * event_dt), Event

        self.accel_gravity = accel_gravity

        # Schedule first gravity event
        heapq.heappush(self.eventQ, (timestep_gravity, 1, GravityEvent(timestep_gravity, accel_gravity)))

        self.start_of_window = 0
        self.end_of_window = self.R
        self.eventQ_at_start = [(t, tidx, e) for t, tidx, e in self.eventQ]  # Copies event queue

        self.particles: list[Particle] = [Particle(collision_stiffness, penalty_layer_thickness,
                                                   particle_positions[i], particle_velocities[i],
                                                   0, particle_masses[i]) for i in range(len(particle_positions))]
        self.walls: list[Collidable] = [Collidable(collision_stiffness,
                                                          penalty_layer_thickness,
                                                          p) for p in wall_positions]
        self.collideables: list[Collidable] = self.walls + self.particles

        self.do_compute_energy = compute_energy  # Should energy be tracked at each snapshot
        self.gravity_base = gravity_energy_base  # Vertical position of "ground" relative to which to compute gravitational energy

        self.sim_time = 0  # Tracks the time of the latest processed event

    def run_sim(self):
        prev_t = None
        prev_t_store = None
        while True:
            while self.eventQ[0][0] <= self.end_of_window:
                te, tidx, e = heapq.heappop(self.eventQ)  # time, event
                #print(f"First: {e}")

                try:
                    assert prev_t is None or te >= prev_t
                except AssertionError as err:
                    print(err)
                    print(te - prev_t, prev_t, te)
                prev_t = te

                self.sim_time = te

                for p in self.particles:
                    p0 = p.current_snapshots[-1]

                    x1 = p0.x + ((te - p0.t) * p0.v)  # integrate position up to this event
                    t1 = te  # update to new time
                    #print(f"Second: {e}")

                    p.current_snapshots.append(Snapshot(x1, -1, t1, None, False, None, None, None, e.get_identifier()))  # Add new snapshot with placeholder v and has_velocity_changed

                pop_snapshot_particle_idx = []  # idx of particles for which this event doesn't act, so we'll pop the snapshot to not bloat the log
                for pi in range(len(self.particles)):
                    p = self.particles[pi]
                    force = e.get_force(p)

                    p0 = p.current_snapshots[-2]
                    pc = p.current_snapshots[-1]
                    if force != 0:
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
                    else:
                        pc.energy = p0.energy
                        pc.kinetic_energy = p0.kinetic_energy
                        pc.potential_energy = p0.potential_energy
                        pc.penalty_energy = p0.penalty_energy
                        pc.v = p0.v
                        pop_snapshot_particle_idx.append(pi)

                if len(pop_snapshot_particle_idx) == len(self.particles):  # If this event didn't affect anyone, clear those snapshots
                    for ii in pop_snapshot_particle_idx:
                        self.particles[ii].current_snapshots.pop(-1)

                # Event-granular log: latest snapshot of all particles
                else:
                    self.logger.record_event([p.current_snapshots[-1] for p in self.particles])

                heapq.heappush(self.eventQ, ((tidx + 1) * e.h, tidx + 1, e))  # Schedule next force event

                self.past_events.append((te, tidx, e))  # Record processed event

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

            # Checks missed collisions
            next_penalty_candidates: list[Collidable] = []  # Objects whose next penalty layer needs activating
            for c in self.collideables:  # Checking c's penalty layers
                move_on = False  # Used as a "break" flag to exit loops once c has been added to the list
                for p in self.particles:  # Moving particles which may have entered a new penalty layer for c
                    if p is not c:  # Don't detect against self, duh

                        # All snapshots where p or c had a velocity change (guaranteed to also include the first and snapshot
                        # of this rollback window)
                        velocity_changed_timestamps = sorted(
                                c.get_velocity_changed_timestamps(self.start_of_window,
                                                                  self.end_of_window).union(
                                p.get_velocity_changed_timestamps(self.start_of_window, self.end_of_window)))

                        # Loop through intervals between above snapshots
                        for i in range(len(velocity_changed_timestamps) - 1):
                            # Get thickness of c's next inactive penalty layer
                            c_next_thickness = c.get_lth_thickness(len(c.active_penalties) + 1)

                            # interval time bounds
                            t0 = velocity_changed_timestamps[i]
                            t1 = velocity_changed_timestamps[i+1]

                            c_positions = [c.get_pos_at_time(t0), c.get_pos_at_time(t1)]
                            p_positions = [p.get_pos_at_time(t0), p.get_pos_at_time(t1)]

                            # Check if particle is already within the penalty layer at start of this interval
                            if c_positions[0] - c_next_thickness <= p_positions[0] <= c_positions[0] + c_next_thickness:
                                next_penalty_candidates.append(c)

                            # Check if there's a t0 <= t <= t1 for which the particle enters the penalty layer
                            else:
                                # Solving two linear equations
                                denom = (c_positions[1] - c_positions[0]) - (p_positions[1] - p_positions[0])
                                if denom != 0:
                                    collision1_t = (p_positions[0] - c_positions[0] - c_next_thickness) / denom
                                    collision2_t = (p_positions[0] - c_positions[0] + c_next_thickness) / denom

                                    if 0 <= collision1_t <= 1 or 0 <= collision2_t <= 1:
                                        next_penalty_candidates.append(c)
                                        move_on = True
                            if move_on: break
                        if move_on: break

            # Initiate rollback
            if len(next_penalty_candidates) > 0:
                # Rollback
                for p in self.particles:
                    p.current_snapshots = [p.current_snapshots[0]]  # Reset snapshots
                self.eventQ = [(t, tidx, e) for t, tidx, e in self.eventQ_at_start]  # Restore initial event queue

                # Engage new penalty layers
                for c in next_penalty_candidates:
                    penalty = PenaltyEvent(self.penalty_timestep, c, len(c.active_penalties) + 1)
                    next_idx = get_next_index(0, penalty.h, self.start_of_window)
                    heapq.heappush(self.eventQ, (next_idx * penalty.h, next_idx, penalty))
                    c.active_penalties.append(penalty)

                    # Assert that these new layers exert 0 force on all particles
                    for p in self.particles:
                        if p is not c:
                            assert penalty.get_force(p) == 0
                #heapq.heapify(self.eventQ)

                # Update saved starting event queue
                self.eventQ_at_start = [(t, tidx, e) for t, tidx, e in self.eventQ]
                prev_t = prev_t_store

                # Notify logger that rollback has occurred
                self.logger.record_rollback()

                yield YieldType.TIME_WINDOW_FINISHED_ROLLBACK, self

            # Proceed to next rollback window
            else:
                for p in self.particles:
                    p.current_snapshots = [p.current_snapshots[-1]]

                # Remove penalties
                for c in self.collideables:
                    innermost_colliding_layer = None

                    # Finds index of c's innermost layer which is exerting a force on at least one particle
                    for i in range(len(c.active_penalties) - 1,  -1, -1):  # Loop backwards through active penalties
                        anyone_colliding = False

                        for p in self.particles:
                            if c.active_penalties[i].get_force(p) != 0:
                                anyone_colliding = True
                                break

                        if anyone_colliding:
                            innermost_colliding_layer = i
                            break

                    def remove_penalties_from_eventQ(penalties):
                        tups_to_remove = []
                        for tup in self.eventQ:
                            if tup[2] in penalties:
                                tups_to_remove.append(tup)
                        for tup in tups_to_remove:
                            self.eventQ.remove(tup)
                        heapq.heapify(self.eventQ)

                    # No layer is exerting force, remove all
                    if innermost_colliding_layer is None:
                        remove_penalties_from_eventQ(c.active_penalties)
                        c.active_penalties = []
                    # Innermost layer i is exerting force. Keep i and all layers further out than i. Delete the rest
                    else:
                        remove_penalties_from_eventQ(c.active_penalties[innermost_colliding_layer+1:])
                        c.active_penalties = c.active_penalties[:innermost_colliding_layer+1]

                # Update bounds to next window and save event queue state
                self.start_of_window = self.end_of_window
                self.end_of_window = self.start_of_window + self.R
                self.eventQ_at_start = [(t, tidx, e) for t, tidx, e in self.eventQ]
                prev_t_store = prev_t

                yield YieldType.TIME_WINDOW_FINISHED, self

    def output_log_data(self): self.logger.output_data()
    def get_collideables(self) -> list[Collidable]: return self.collideables
    def get_particles(self) -> list[Particle]: return self.particles
    def get_walls(self) -> list[Collidable]: return self.walls
    def get_window_size(self) -> float: return self.R
    def get_eventQ(self) -> list[tuple[float, int, Event]]: return self.eventQ
    def get_past_events(self) -> list[tuple[float, int, Event]]: return self.past_events
    def get_logger(self) -> Logger: return self.logger
    def get_sim_time(self) -> float: return self.sim_time

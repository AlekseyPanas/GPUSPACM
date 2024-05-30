from __future__ import  annotations
import pygame
from matplotlib import pyplot
import math
import numpy
from dataclasses import dataclass
from typing import Optional
import heapq
from abc import abstractmethod


"""
Steps:
- Start sim with params
- Schedule first gravity force event
- Record first time window data
- While events in queue within this time interval R:
    - Pop event
    - Integrate position and update vertex ti, xi
    - Integrate force
    - Schedule next force event if applicable
- Advance to end of time window if space left
- Check missed collisions within time window
- If missed
    - Rollback
    - For each missed collision where object A entered B's next inactive penalty layer
        - Enable B's next penalty layer by scheduling the first force event affecting all objects
- Else
    - Remove any penalty layers exerting 0 force at the end of the time window (detect this by 
    plugging in the position values at the end of the time window into the penalty force equation and seeing if its 0)
"""


@dataclass
class Snapshot:
    x: float
    v: float
    t: float
    energy: Optional[float]
    has_velocity_changed: bool


class Event:
    """ Recurring event applying force to all particles in the simulation """
    def __init__(self, timestep: float):
        self.h = timestep

    @abstractmethod
    def get_force(self, target: Particle):
        """Return the force, in newtons, that this even applies on the given particle"""


class GravityEvent(Event):
    """ Force of Gravity """
    def __init__(self, timestep: float, accel_gravity: float):
        super().__init__(timestep)
        self.g = accel_gravity

    def get_force(self, target: Particle) -> float:
        return target.mass * self.g


class PenaltyEvent(Event):
    """ Single collision penalty layer for a collideable object """
    def __init__(self, timestep: float, owner: Collidable, layer: int):
        super().__init__(timestep)
        self.owner = owner
        self.layer = layer

    def get_force(self, target: Particle) -> float:
        targ_x = target.get_latest_position()
        own_x = self.owner.get_latest_position()

        gap = abs(own_x - targ_x) - self.owner.get_lth_thickness(self.layer)
        if gap > 0:
            return 0
        else:
            return self.owner.get_lth_stiffness(self.layer) * gap * (-1 if targ_x > own_x else 1)


class Collidable:
    def __init__(self, collision_stiffness: float, first_layer_thickness: float, x: float):
        self.r1 = collision_stiffness  # first penalty layer's collision stiffness
        self.n1 = first_layer_thickness
        self.init_x = x
        self.active_penalties: list[PenaltyEvent] = []

    def get_lth_thickness(self, l: int) -> float:
        return self.n1 * (l ** (-0.25))  # ACM paper section 4

    def get_lth_stiffness(self, l: int) -> float:
        return self.r1 * (l ** 3)  # ACM paper section 4

    def get_velocity_changed_snapshots(self, window_start: float, window_end: float) -> set[Snapshot]:
        """
        When detecting collisions across a rollback window, we need the snapshots where velocity of the object
        changed to compute piecewise linear collision intervals
        :return: a list of velocity-changed snapshots including window_start and window_end by default
        """
        return {Snapshot(self.init_x, 0, window_start, None, False),
                Snapshot(self.init_x, 0, window_end, None, False)}

    def get_latest_position(self) -> float:
        return self.init_x

    def get_pos_at_time(self, t):
        return self.init_x


class Particle(Collidable):
    """
    current_snapshots: snapshots in the current rollback window. First snapshot in this list is the start of
    this rollback window
    previous_snapshots: rollback windows which have passed.
    """
    def __init__(self, collision_stiffness: float, first_layer_thickness: float,
                 x: float, v: float, cur_time: float, mass: float):
        super().__init__(collision_stiffness, first_layer_thickness, x)

        self.previous_snapshots: list[Snapshot] = []
        self.current_snapshots: list[Snapshot] = []

        self.current_snapshots.append(Snapshot(x, v, cur_time, None, True))

        self.mass = mass

    def get_velocity_changed_snapshots(self, window_start: float, window_end: float) -> set[Snapshot]:
        return {self.current_snapshots[0], self.current_snapshots[-1]}.union(
            {snap for snap in self.current_snapshots if snap.has_velocity_changed})

    def get_latest_position(self) -> float:
        return self.current_snapshots[-1].x

    def get_pos_at_time(self, t) -> float:
        # assumes snapshots are ordered temporally in their list which is a consequence of the code
        before = None
        for snap in self.current_snapshots:
            if snap.t == t:
                return snap.x
            elif snap.t < t:
                before = snap
            else:
                break
        return before.x + ((t - before.t) * before.v)


class SPACM1DSim:
    def __init__(self, rollback_window_size: float, accel_gravity: float, timestep_gravity: float,
                 particle_positions: list[float], particle_velocities: list[float], wall_positions: list[float],
                 particle_masses: list[float], collision_stiffness: float, penalty_layer_thickness: float,
                 penalty_timestep: float):
        assert len(particle_positions) == len(particle_velocities)

        self.R = rollback_window_size
        self.penalty_timestep = penalty_timestep

        self.eventQ: list[tuple[float, Event]] = []

        # Schedule first gravity event
        heapq.heappush(self.eventQ, (timestep_gravity, GravityEvent(timestep_gravity, accel_gravity)))

        self.start_of_window = 0
        self.end_of_window = self.R
        self.eventQ_at_start = [(t, e) for t, e in self.eventQ]  # Copies event queue

        self.particles: list[Particle] = [Particle(collision_stiffness, penalty_layer_thickness,
                                                   particle_positions[i], particle_velocities[i],
                                                   0, particle_masses[i]) for i in range(len(particle_positions))]
        self.collideables: list[Collidable] = [Collidable(collision_stiffness,
                                                          penalty_layer_thickness,
                                                          p) for p in wall_positions] + self.particles

    def run_sim(self):
        while True:
            while self.eventQ[0][0] <= self.end_of_window:
                te, e = heapq.heappop(self.eventQ)  # time, event

                for p in self.particles:
                    p0 = p.current_snapshots[-1]

                    x1 = p0.x + ((te - p0.t) * p0.v)  # integrate position up to this event
                    t1 = te  # update to new time

                    p.current_snapshots.append(Snapshot(x1, -1, t1, None, False))  # Add new snapshot with placeholder v and has_velocity_changed

                for p in self.particles:
                    p0 = p.current_snapshots[-1]
                    v1 = p0.v + e.h * (e.get_force(p) / p.mass)  # integrate force with respect to force's timestep

                    # Update placeholder v and has_velocity_changed values
                    p0.has_velocity_changed = p0.v != v1
                    p0.v = v1

                heapq.heappush(self.eventQ, (te + e.h, e))  # Schedule next force event

            # Linearly update particle positions and timestamps to the end of the rollback window
            for p in self.particles:
                p0 = p.current_snapshots[-1]
                if p0.t < self.end_of_window:
                    x1 = p0.x + ((self.end_of_window - p0.t) * p0.v)
                    t1 = self.end_of_window
                    p.current_snapshots.append(Snapshot(x1, p0.v, t1, None, False))

            # Checks missed collisions
            next_penalty_candidates: list[Collidable] = []  # Objects whose next penalty layer needs activating
            for c in self.collideables:  # Checking c's penalty layers
                for p in self.particles:  # Moving particles which may have entered a new penalty layer for c

                    # All snapshots where p or c had a velocity change (guaranteed to also include the first and snapshot
                    # of this rollback window)
                    velocity_changed_snaps = sorted(
                            c.get_velocity_changed_snapshots(self.start_of_window,
                                                                                  self.end_of_window).union(
                            p.get_velocity_changed_snapshots(self.start_of_window, self.end_of_window)),
                        key=lambda snap: snap.t
                    )

                    # Loop through intervals between above snapshots
                    for i in range(len(velocity_changed_snaps) - 1):
                        # Get thickness of c's next inactive penalty layer
                        c_next_thickness = c.get_lth_thickness(len(c.active_penalties) + 1)

                        # interval time bounds
                        t0 = velocity_changed_snaps[i].t
                        t1 = velocity_changed_snaps[i+1].t

                        c_positions = [c.get_pos_at_time(t0), c.get_pos_at_time(t1)]
                        p_positions = [c.get_pos_at_time(t0), c.get_pos_at_time(t1)]

                        # Check if particle is already within the penalty layer at start of this interval
                        if c_positions[0] - c_next_thickness <= p_positions[0] <= c_positions[0] + c_next_thickness:
                            next_penalty_candidates.append(c)

                        # Check if there's a t0 <= t <= t1 for which the particle enters the penalty layer
                        else:
                            # Solving two linear equations
                            denom = ((c_positions[1] - c_positions[0]) / (t1 - t0)) - \
                                    ((c_positions[1] - c_positions[0]) / (t1 - t0))
                            if denom != 0:
                                collision1_t = (p_positions[0] - c_positions[0] - c_next_thickness) / denom
                                collision2_t = (p_positions[0] - c_positions[0] + c_next_thickness) / denom

                                if t0 <= collision1_t <= t1 or t0 <= collision2_t <= t1:
                                    next_penalty_candidates.append(c)

            # Initiate rollback
            if len(next_penalty_candidates) > 0:
                # Rollback
                for p in self.particles:
                    p.current_snapshots = [p.current_snapshots[0]]  # Reset snapshots
                    self.eventQ = [(t, e) for t, e in self.eventQ_at_start]  # Restore initial event queue

                # Engage new penalty layers
                for c in next_penalty_candidates:
                    penalty = PenaltyEvent(self.penalty_timestep, c, len(c.active_penalties) + 1)
                    self.eventQ.append((self.start_of_window + penalty.h, penalty))
                    c.active_penalties.append(penalty)

            # Proceed to next rollback window
            else:
                # Move old window's snapshots into history
                for p in self.particles:
                    p.previous_snapshots += p.current_snapshots[:-1]
                    p.current_snapshots = p.current_snapshots[-1]

                # Update bounds to next window and save event queue state
                self.start_of_window = self.end_of_window
                self.end_of_window = self.start_of_window + self.R
                self.eventQ_at_start = [(t, e) for t, e in self.eventQ]

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
                            if tup[1] in penalties:
                                tups_to_remove.append(tup)
                        for tup in tups_to_remove:
                            self.eventQ.remove(tup)
                        self.eventQ = heapq.heapify(self.eventQ)

                    # No layer is exerting force, remove all
                    if innermost_colliding_layer is None:
                        remove_penalties_from_eventQ(c.active_penalties)
                        c.active_penalties = []
                    # Innermost layer i is exerting force. Keep i and all layers further out than i. Delete the rest
                    else:
                        remove_penalties_from_eventQ(c.active_penalties[innermost_colliding_layer+1:])
                        c.active_penalties = c.active_penalties[:innermost_colliding_layer+1]

            yield self


# class PygameVisualizer:
#     def __init__(self):
#         self.screen = None
#
#     def

from __future__ import annotations

from enum import IntEnum
from typing import Optional
from dataclasses import dataclass
from abc import abstractmethod
from z_logging.loggers import Logger
import math


class YieldType(IntEnum):
    EVENT_FINISHED = 0
    TIME_WINDOW_FINISHED = 1
    TIME_WINDOW_FINISHED_ROLLBACK = 2


@dataclass(unsafe_hash=True)
class Snapshot:
    x: float
    v: float
    t: float
    energy: Optional[float]
    has_velocity_changed: bool
    kinetic_energy: Optional[float]
    potential_energy: Optional[float]
    penalty_energy: Optional[float]
    event_identifier: Optional[int]  # Name of the event which triggered this snapshot, or None if the snapshot is not due to an event


def get_next_time(start_time: float, dt: float, cur_time: float):
    """
    :param start_time: time when this clock started ticking
    :param dt: timestep of globally ticking clock
    :param cur_time: the current timestamp
    :return: the time of the next tick of a global clock that ticked from start_time at intervals of dt. for example,
    get_next_time(0, 3, 7) returns 9 because the global clock ticks at 3, 6, 9, ..., and 9 is the next tick after 7.
    the returned timestamp excludes cur_time, so get_next_time(0, 2, 4) returns 6 and not 4
    """
    return start_time + ((((cur_time - start_time) // dt) + 1) * dt)


class Event:
    """ Recurring event applying force to all particles in the simulation """
    def __init__(self, timestep: float):
        self.h = timestep

    @abstractmethod
    def get_force(self, target: Particle):
        """Return the force, in newtons, that this even applies on the given particle"""

    def __lt__(self, other):
        """Completely arbitrary comparison so that code doesn't crash"""
        return self.h < other.h

    @abstractmethod
    def get_identifier(self) -> int:
        """Return string name of this event to use as a logging identifier"""


class GravityEvent(Event):
    """ Force of Gravity """
    def __init__(self, timestep: float, accel_gravity: float):
        super().__init__(timestep)
        self.g = accel_gravity

    def get_force(self, target: Particle) -> float:
        return target.mass * self.g

    def get_identifier(self) -> int:
        return 0


class PenaltyEvent(Event):
    """ Single collision penalty layer for a collideable object """
    def __init__(self, base_timestep: float, owner: Collidable, layer: int):
        super().__init__(owner.get_lth_timestep(layer, base_timestep))
        self.owner = owner
        self.layer = layer

    def get_gap(self, q: float):
        """Return gap function g(q) for a particle position q"""
        return abs(self.owner.get_latest_position() - q) - self.owner.get_lth_thickness(self.layer)

    def get_force(self, target: Particle) -> float:
        if target is self.owner: return 0

        # the latest position is guaranteed to be at an equal timestep because SPACM sim code always updates
        # positions of EVERY particle before computing forces. If stencils are ever implemented this may no
        # longer be the case
        targ_x = target.get_latest_position()
        own_x = self.owner.get_latest_position()

        gap = self.get_gap(targ_x)
        if gap > 0:
            return 0
        else:
            return self.owner.get_lth_stiffness(self.layer) * gap * (-1 if targ_x > own_x else 1)

    def get_identifier(self) -> int:
        return self.layer


class Collidable:
    next_id = 0

    def __init__(self, collision_stiffness: float, first_layer_thickness: float, x: float):
        self.r1 = collision_stiffness  # first penalty layer's collision stiffness
        self.n1 = first_layer_thickness
        self.init_x = x
        self.active_penalties: list[PenaltyEvent] = []
        self.obj_id = Collidable.next_id
        Collidable.next_id += 1

    def get_lth_thickness(self, l: int) -> float:
        return self.n1 * (l ** (-0.25))  # ACM paper section 4

    def get_lth_stiffness(self, l: int) -> float:
        return self.r1 * (l ** 3)  # ACM paper section 4

    def get_lth_timestep(self, l: int, base_dt: float, fudge=1e-4) -> float:
        return base_dt / l / math.sqrt(l + fudge)

    def get_velocity_changed_timestamps(self, window_start: float, window_end: float) -> set[float]:
        """
        When detecting collisions across a rollback window, we need the snapshots where velocity of the object
        changed to compute piecewise linear collision intervals
        :return: a list of velocity-changed snapshots including window_start and window_end by default
        """
        return {window_start, window_end}

    def get_latest_position(self) -> float:
        return self.init_x

    def get_pos_at_time(self, t):
        return self.init_x


class Particle(Collidable):
    """
    current_snapshots: snapshots in the current rollback window. First snapshot in this list is the start of
    this rollback window
    """
    def __init__(self, collision_stiffness: float, first_layer_thickness: float,
                 x: float, v: float, cur_time: float, mass: float):
        super().__init__(collision_stiffness, first_layer_thickness, x)

        self.current_snapshots: list[Snapshot] = []

        self.current_snapshots.append(Snapshot(x, v, cur_time, None, True, None, None, None, None))

        self.mass = mass

    def get_velocity_changed_timestamps(self, window_start: float, window_end: float) -> set[float]:
        return {self.current_snapshots[0].t, self.current_snapshots[-1].t}.union(
            {snap.t for snap in self.current_snapshots if snap.has_velocity_changed})

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


class Sim:
    @abstractmethod
    def run_sim(self):
        """Run the simulation and yield one of the types in YieldType whenever a loop has been executed (e.g yield
        on every event, every rollback, and every finished window)"""

    @abstractmethod
    def output_log_data(self):
        """Tell the simulator (and underlying logger) to output any cached log data to disc. This method is typically
        called on exit"""

    @abstractmethod
    def get_collideables(self) -> list[Collidable]:
        """Get all collideables"""

    @abstractmethod
    def get_particles(self) -> list[Particle]:
        """Get all collideables which are particles"""

    @abstractmethod
    def get_walls(self) -> list[Collidable]:
        """Get all collideables which aren't particles"""

    @abstractmethod
    def get_window_size(self) -> float:
        """Get size of the rollback window, R"""

    @abstractmethod
    def get_eventQ(self) -> list[tuple[float, Event]]:
        """Get all upcoming events"""

    @abstractmethod
    def get_past_events(self) -> list[tuple[float, Event]]:
        """Get all events which have already been executed"""

    @abstractmethod
    def get_logger(self) -> Logger:
        """Get the underlying logger used by this simulation to record data"""

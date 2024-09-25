from __future__ import annotations
import z_sims.core.objects as objs
from abc import abstractmethod


class Event:
    """ Recurring event applying force to all particles in the simulation """
    def __init__(self, timestep: float):
        self.h = timestep

    @abstractmethod
    def get_force(self, target: objs.Particle):
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

    def get_force(self, target: objs.Particle) -> float:
        return target.mass * self.g

    def get_identifier(self) -> int:
        return 0


class PenaltyEvent(Event):
    """ Single collision penalty layer for a collideable object """
    def __init__(self, base_timestep: float, owner: objs.Collidable, layer: int):
        super().__init__(owner.get_lth_timestep(layer, base_timestep))
        self.owner = owner
        self.layer = layer

    def get_gap(self, q: float):
        """Return gap function g(q) for a particle position q"""
        return abs(self.owner.get_latest_position() - q) - self.owner.get_lth_thickness(self.layer)

    def get_force(self, target: objs.Particle) -> float:
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
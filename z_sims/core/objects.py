from __future__ import annotations
from z_sims.core.events import PenaltyEvent
from z_sims.core.data import Snapshot


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
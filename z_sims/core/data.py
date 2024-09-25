from __future__ import annotations
from typing import Optional
from enum import IntEnum
from dataclasses import dataclass


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

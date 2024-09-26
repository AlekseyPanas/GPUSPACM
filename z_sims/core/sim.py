from __future__ import annotations
from abc import abstractmethod
from z_sims.core.objects import Collidable, Particle
from z_sims.core.events import Event
from z_logging.loggers import Logger


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

    @abstractmethod
    def get_sim_time(self) -> float:
        """Get the time of the simulation currently (latest event processed)"""

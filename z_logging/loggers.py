from __future__ import annotations
from abc import abstractmethod
from z_sims.core.data import Snapshot
from npy_append_array import NpyAppendArray
import numpy as np
import os
from datetime import datetime


class Logger:
    @abstractmethod
    def record_config(self, params: dict):
        """Provides a locals() object of the simulation parameters in __init__. Use this to record the configuration
        of a given experiment"""

    @abstractmethod
    def record_event(self, t: float, total_energy: float, event_id: int, num_particles: int, snapshots: list[tuple[int, Snapshot]]):
        """Record a processed event. This includes the event time, the total energy after this event is processed,
        the event identifier as decided by the sim, and a list of particle snapshots (paired with their index) of
        particles affected by this event. The number of total particles is also provided"""

    @abstractmethod
    def record_window_catchup(self, t: float, total_energy: float, snapshots: list[Snapshot]):
        """Record the state of ALL particles at the end of a time window. This fires immediately after all events
        for the time window have been processed BEFORE any rollback decision has been made."""

    @abstractmethod
    def record_rollback(self):
        """Record that a rollback has occurred. Subsequent datapoints will start in overlap"""

    @abstractmethod
    def record_window_success(self):
        """Record that the current window succeeded so subsequent datapoints will start in the next window"""

    @abstractmethod
    def output_data(self):
        """Write any cached data to disc. This may be called multiple times in the same simulation"""

    @abstractmethod
    def quit(self):
        """Called when the simulation quits. Use this to clean up anything"""


class NumpyLogger(Logger):
    FOLDER_NAME = "npdat"
    """
    cache_size: This is the number of logs that will be stored in memory before writing to file (a.k.a batch size)
    custom_experiment_prefix: This is a string name to identify the experiment. It will be included in the log filename
    log_root_folder_path: This is a path to the directory where logs are stored. npy data will be writen to this dir /npdat
    """
    def __init__(self, log_root_folder_path: str, do_log: bool, custom_experiment_prefix: str, cache_size=300):
        """cache_size indicates how many records to store before flushing to file"""
        self.num_particles = -1

        self.folder_name = self.FOLDER_NAME
        if self.folder_name not in os.listdir(log_root_folder_path):
            os.mkdir(os.path.join(log_root_folder_path, self.folder_name))

        filename_root = "".join((c if c != ":" else "." for c in f"{custom_experiment_prefix}_{str(datetime.now()).replace(' ', '_')}"))
        filename_main = filename_root + ".npy"
        filename_totalenergy = filename_root + "-totalenergy.npy"
        self.filepath_main = os.path.join(log_root_folder_path, self.folder_name, filename_main)
        self.filepath_totalenergy = os.path.join(log_root_folder_path, self.folder_name, filename_totalenergy)
        self.do_log = do_log
        self.did_init = False

        self.handle_main = None
        self.handle_totalenergy = None
        if self.do_log:
            self.handle_main = NpyAppendArray(self.filepath_main, delete_if_exists=True)
            self.handle_totalenergy = NpyAppendArray(self.filepath_totalenergy, delete_if_exists=True)

        self.cache_main: np.ndarray | None = None
        self.cache_totalenergy: np.ndarray | None = None

        self.cache_size = cache_size
        self.cache_counter = 0

    def __init_array(self):
        self.cache = np.ndarray([0, self.num_particles, 9])

    def __increment_and_flush_cache(self):
        self.cache_counter += 1
        if self.cache_counter >= self.cache_size:
            self.cache_counter = 0
            self.__flush()

    def __flush(self):
        self.handle.append(self.cache)
        self.__init_array()

    def record_config(self, params: dict):
        if not self.do_log: return

        with open(self.filepath + "-config.txt", "w") as file:
            file.write(str(params))

    def record_rollback(self):
        if not self.do_log: return

        self.cache = np.concatenate([self.cache, np.array([[[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0, -1.0] for _ in range(self.num_particles)]])])
        self.__increment_and_flush_cache()

    def record_event(self, snapshots: list[Snapshot], total_energy: float):
        if not self.do_log: return
        if not self.did_init:
            self.did_init = True
            self.num_particles = len(snapshots)
            self.__init_array()

        self.cache = np.concatenate([self.cache, np.array([[
            [snap.x, snap.v, snap.t, snap.energy, snap.kinetic_energy,
             snap.potential_energy, snap.penalty_energy, 0.0, snap.event_identifier, total_energy] for snap in snapshots
        ]])])
        self.__increment_and_flush_cache()

    def record_window_snapshots(self, snapshots: list[Snapshot], total_energy: float):
        if not self.do_log: return

        new_dat = np.array([[
            [snap.x, snap.v, snap.t, snap.energy, snap.kinetic_energy,
             snap.potential_energy, snap.penalty_energy, 1.0, -1.0, total_energy] for snap in snapshots
        ]])
        self.cache = np.concatenate([self.cache, new_dat])
        self.__increment_and_flush_cache()

    def output_data(self):
        self.__flush()

    def quit(self):
        self.__flush()
        self.handle.close()

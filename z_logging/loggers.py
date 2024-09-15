from __future__ import annotations
from abc import abstractmethod
from z_sims.core import Snapshot
from npy_append_array import NpyAppendArray
import numpy as np
import os
from datetime import datetime


class Logger:
    @abstractmethod
    def rollback(self):
        """Notifies the logger that a rollback has occurred. Subsequent datapoints will start in overlap"""

    @abstractmethod
    def record_config(self, params: dict):
        """Provides a locals() object of the simulation parameters in __init__. Use this to record the configuration
        of a given experiment"""

    @abstractmethod
    def record_snapshots(self, snapshots: list[Snapshot]):
        """Record a new list of snapshots for every particle. The particle order in the given snapshots list is
        maintained across all calls"""

    @abstractmethod
    def record_window_snapshots(self, snapshots: list[Snapshot]):
        """Record a new list of snapshots for every particle. The particle order in the given snapshots list is
        maintained across all calls. Unlike record_snapshots, this method is only called once at the end of a
        successful rollback window. Useful in case you dont care about logging each rollback"""

    @abstractmethod
    def output_data(self):
        """Output current data. This may be called multiple times in the same simulation"""

    @abstractmethod
    def quit(self):
        """Called when the simulation quits. Use this to clean up anything"""


class NumpyLogger(Logger):
    def __init__(self, do_log: bool, custom_experiment_prefix: str, cache_size=300):
        """cache_size indicates how many records to store before flushing to file"""
        self.dat: np.ndarray | None = None
        self.num_particles = -1

        self.folder_name = "npdat"
        if self.folder_name not in os.listdir("."):
            os.mkdir(f"./{self.folder_name}")

        self.filepath = os.path.join(self.folder_name, "".join((c if c != ":" else "." for c in f"{custom_experiment_prefix}_{str(datetime.now()).replace(' ', '_')}.npy")))
        self.do_log = do_log
        self.did_init = False

        self.handle = None
        if self.do_log:
            self.handle = NpyAppendArray(self.filepath, delete_if_exists=True)

        self.cache: np.ndarray = np.array([])

        self.cache_size = cache_size
        self.cache_counter = 0

    def __init_array(self, num_particles):
        self.dat = np.ndarray([0, num_particles, 8])
        self.num_particles = num_particles

    def __increment_and_flush_cache(self):
        self.cache_counter += 1
        if self.cache_counter >= self.cache_size:
            self.cache_counter = 0
            self.handle.append(self.cache)
            self.cache = np.array([])

    def record_config(self, params: dict):
        if not self.do_log: return

        with open(self.filepath + "-config.txt", "w") as file:
            file.write(str(params))

    def rollback(self):
        if not self.do_log: return

        self.cache = np.concatenate([self.cache, np.array([[[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0, -1.0] for _ in range(self.num_particles)]])])
        self.__increment_and_flush_cache()

    def record_snapshots(self, snapshots: list[Snapshot]):
        if not self.do_log: return
        if not self.did_init:
            self.did_init = True
            self.__init_array(len(snapshots))

        self.cache = np.concatenate([self.cache, np.array([[
            [snap.x, snap.v, snap.t, snap.energy, snap.kinetic_energy,
             snap.potential_energy, snap.penalty_energy, 0.0, snap.event_identifier] for snap in snapshots
        ]])])
        self.__increment_and_flush_cache()

    def record_window_snapshots(self, snapshots: list[Snapshot]):
        if not self.do_log: return

        self.cache = np.concatenate([self.cache, np.array([[
            [snap.x, snap.v, snap.t, snap.energy, snap.kinetic_energy,
             snap.potential_energy, snap.penalty_energy, 1.0, -1.0] for snap in snapshots
        ]])])
        self.__increment_and_flush_cache()

    def output_data(self):
        pass

    def quit(self):
        self.handle.close()

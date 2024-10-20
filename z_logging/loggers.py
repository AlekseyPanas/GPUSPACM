from __future__ import annotations
from abc import abstractmethod
from z_sims.core.data import Snapshot
from npy_append_array import NpyAppendArray
import numpy as np
import os
from datetime import datetime


class Logger:
    @abstractmethod
    def record_config(self, params: dict, num_particles: int):
        """Provides a locals() object of the simulation parameters in __init__. Use this to record the configuration
        of a given experiment. This function is called once before all other logging so initialization can be done here"""

    @abstractmethod
    def record_event(self, window_idx: int, t: float, total_energy: float, event_id: int, num_particles: int, snapshots: list[tuple[int, Snapshot]]):
        """Record a processed event. This includes the event time, the total energy after this event is processed,
        the event identifier as decided by the sim, and a list of particle snapshots (paired with their index) of
        particles affected by this event. The number of total particles is also provided"""

    @abstractmethod
    def record_window_catchup(self, window_idx: int, t: float, total_energy: float, snapshots: list[Snapshot]):
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
    """
    @param cache_size: This is the number of logs that will be stored in memory before writing to file (a.k.a batch size)
    @param custom_experiment_prefix: This is a string name to identify the experiment. It will be included in the log filename
    @param log_root_folder_path: This is a path to the directory where logs are stored. npy data will be writen to this dir /npdat

    Event ID types:
        >= 0: Event ID as assigned by simulator
        -1: End of window catch up
        -2: Rollback
        -3: End of window success
    """

    FOLDER_NAME = "npdat"

    def __init__(self, log_root_folder_path: str, do_log: bool, custom_experiment_prefix: str, cache_size=300):
        """cache_size indicates how many records to store before flushing to file"""
        self.num_particles = -1

        # Create outer folder if not there
        self.folder_name = self.FOLDER_NAME
        self.folder_path = os.path.join(log_root_folder_path, self.folder_name)
        if self.folder_name not in os.listdir(log_root_folder_path):
            os.mkdir(self.folder_path)

        # Create subfolder if not there
        self.subfolder_name = "".join((c if c != ":" else "." for c in f"{custom_experiment_prefix}_{str(datetime.now()).replace(' ', '_')}"))
        self.subfolder_path = os.path.join(self.folder_path, self.subfolder_name)
        if self.subfolder_name not in os.listdir(self.folder_path):
            os.mkdir(self.subfolder_path)

        # Store file handles for each particle's individual file as well as the global event file
        self.particle_handles = []
        self.particle_filenames = []
        self.particle_filepaths = []
        self.particle_caches: list[np.ndarray] = []
        self.event_handle = None
        self.event_cache: np.ndarray | None = None

        self.event_cur_idx = -1  # Tracks how many events were added thus far (idx)

        self.do_log = do_log

        # Track the cache to flush to file
        self.cache_size = cache_size
        self.cache_counter = 0

    def __reset_cache(self):
        """Clear all caches"""
        self.particle_caches = [np.ndarray([0, 7]) for _ in range(self.num_particles)]
        self.event_cache = np.ndarray([0, 4])

    def __flush(self):
        """Flush all caches to file and clear the caches"""
        for i in range(self.num_particles):
            self.particle_handles[i].append(self.particle_caches[i])
        self.event_handle.append(self.event_cache)
        self.__reset_cache()

    def __increment_and_flush_cache(self):
        """Increment the cache counter and flush if cache size is maxed out"""
        self.cache_counter += 1
        if self.cache_counter >= self.cache_size:
            self.cache_counter = 0
            self.__flush()

    def record_config(self, params: dict, num_particles: int):
        if not self.do_log: return

        self.num_particles = num_particles
        self.particle_filenames = [self.subfolder_name + f"-particle-{i}.npy" for i in range(num_particles)]
        self.particle_filepaths = [os.path.join(self.subfolder_path, filename) for filename in self.particle_filenames]
        self.particle_handles = [NpyAppendArray(filepath, delete_if_exists=True) for filepath in self.particle_filepaths]
        self.event_handle = NpyAppendArray(os.path.join(self.subfolder_path, self.subfolder_name + "-events.npy"), delete_if_exists=True)
        self.__reset_cache()

        with open(os.path.join(self.subfolder_path, self.subfolder_name + "-config.txt"), "w") as file:
            file.write(str(params))

    def record_event(self, window_idx: int, t: float, total_energy: float, event_id: int, num_particles: int,
                     snapshots: list[tuple[int, Snapshot]]):
        if not self.do_log: return

        # Add next event entry
        self.event_cache = np.concatenate([self.event_cache, np.array([[event_id, t, total_energy, window_idx]])])
        self.event_cur_idx += 1

        # For relevant particles, log particle snapshot into the particle's file
        for pi, snap in snapshots:
            self.particle_caches[pi] = np.concatenate([self.particle_caches[pi],
                                                       np.array([[self.event_cur_idx, snap.x, snap.v, snap.energy, snap.kinetic_energy, snap.potential_energy, snap.penalty_energy]])])

        self.__increment_and_flush_cache()

    def record_window_catchup(self, window_idx: int, t: float, total_energy: float, snapshots: list[Snapshot]):
        if not self.do_log: return

        # Add next event entry
        self.event_cache = np.concatenate([self.event_cache, np.array([[-1, t, total_energy, window_idx]])])
        self.event_cur_idx += 1

        for pi in range(len(snapshots)):
            snap = snapshots[pi]
            self.particle_caches[pi] = np.concatenate([self.particle_caches[pi],
                                                       np.array([[self.event_cur_idx, snap.x, snap.v, snap.energy, snap.kinetic_energy, snap.potential_energy, snap.penalty_energy]])])

        self.__increment_and_flush_cache()

    def record_rollback(self):
        if not self.do_log: return

        self.event_cache = np.concatenate([self.event_cache, np.array([[-2, -1, -1, -1]])])
        self.event_cur_idx += 1
        self.__increment_and_flush_cache()

    def record_window_success(self):
        if not self.do_log: return

        self.event_cache = np.concatenate([self.event_cache, np.array([[-3, -1, -1, -1]])])
        self.event_cur_idx += 1
        self.__increment_and_flush_cache()

    def output_data(self):
        self.__flush()

    def quit(self):
        self.__flush()
        self.event_handle.close()
        for h in self.particle_handles:
            h.close()

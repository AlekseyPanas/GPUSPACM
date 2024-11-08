"""
Simulations save data in npy files. This script provides utilities to read and output that data into
other formats such as txt, tensorboard, or matplotlib
"""
from __future__ import annotations
from tensorboardX import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
from enum import IntEnum
import os
import struct


class EntryType(IntEnum):
    EVENT_ENTRY = 0
    WINDOW_CATCHUP = 1
    ROLLBACK = 2
    WINDOW_SUCCESS = 3


@dataclass(unsafe_hash=True)
class EventEntry:
    t: float
    entry_type: EntryType
    total_energy: float
    event_identifier: int
    window_idx: int


@dataclass(unsafe_hash=True)
class Snapshot:
    """entry_type is either EVENT or CATCHUP. In the case of CATCHUP the event_identifier is -1"""
    x: float
    v: float
    t: float
    energy: Optional[float]
    kinetic_energy: Optional[float]
    potential_energy: Optional[float]
    penalty_energy: Optional[float]
    event_identifier: int
    entry_type: EntryType
    window_idx: int
    event_entry_idx: int


class DataReader:
    """Reads a log file of a specific type and parses it"""
    @abstractmethod
    def get_subfolder_name(self) -> str:
        """Get the name of the experiment folder being parsed (not the whole path, just the file name)"""

    @abstractmethod
    def get_num_particles(self) -> int:
        """Get the number of particles in the simulation that generated the data file being read by this reader"""

    @abstractmethod
    def event_entries(self) -> list[EventEntry]:
        """Return an iterable of all event entries in the simulation"""

    @abstractmethod
    def get_event_entry_at(self, idx) -> EventEntry:
        """Given an index into all the event entries that exist in temporal order, return the event entry at that idx"""

    @abstractmethod
    def get_num_event_entries(self) -> int:
        """Get total number of all event entries (i.e length of event entry data)"""

    @abstractmethod
    def event_granular(self, i) -> list[Snapshot]:
        """Return an iterable of all recorded snapshots of the simulation for particle i"""

    @abstractmethod
    def event_granular_raw(self, i) -> list[tuple]:
        """Return an iterable of all recorded snapshots of the simulation for particle i. The data returned
        is a raw tuple containing unparsed numerical data corresponding to parameters in the Snapshot class"""

    @abstractmethod
    def window_granular(self, i) -> list[Snapshot]:
        """Return iterable of snapshots at the end of successful (non-rollback) time windows
        for particle at index i in the simulation"""

    @abstractmethod
    def window_granular_raw(self, i) -> list[tuple]:
        """Return iterable of snapshots at the end of successful (non-rollback) time windows
        for particle i throughout the simulation. The data returned is a raw tuple containing
        unparsed numerical data corresponding to parameters in the Snapshot class"""


class NumpyDataReader(DataReader):
    """Parses an .npy file with snapshot data. The snapshot data is expected as a 3D array
    where each 'row' is a list of snapshots for each particle. A snapshot is stored as an array
    of values in the order of the Snapshot dataclass"""
    def __init__(self, subfolder_path: str):
        self.subfolder_path = subfolder_path

        self.num_particles: int = len(os.listdir(self.subfolder_path)) - 2  # config and -events file excluded

        self.event_filepath = None
        self.particle_filepaths: list[str | None] = [None] * self.num_particles

        for filename in os.listdir(self.subfolder_path):
            if filename.endswith("-events.npy"):
                self.event_filepath = os.path.join(self.subfolder_path, filename)
            elif not filename.endswith("-config.txt"):
                self.particle_filepaths[int(filename[-5])] = os.path.join(self.subfolder_path, filename)

        assert self.event_filepath is not None

        self.event_dat = np.load(self.event_filepath, mmap_mode="r")
        self.particle_dats = [np.load(path, mmap_mode="r") for path in self.particle_filepaths]

    @staticmethod
    def __id_to_entry(event_id: int) -> EntryType:
        """Given the event_id value from the raw numpy data, convert to the enum value"""
        event_id = int(event_id)

        return {-3: EntryType.WINDOW_SUCCESS, -2: EntryType.ROLLBACK, -1: EntryType.WINDOW_CATCHUP}.get(event_id, EntryType.EVENT_ENTRY)

    def __tuple_from_particle_data_row(self, row) -> tuple:
        """Given a tuple of numpy data for a single entry of a particle snapshot, return a python tuple in the correct
        order for the snapshot datastructure"""
        return row[1], row[2], self.event_dat[int(row[0])][1], row[3], row[4], row[5], row[6], \
            self.event_dat[int(row[0])][0], self.__id_to_entry(self.event_dat[int(row[0])][0]).value, self.event_dat[int(row[0])][3], row[0]

    def __snapshot_from_particle_data_row(self, row) -> Snapshot:
        """Given a tuple of numpy data for a single entry of a particle snapshot, return a snapshot of this data"""
        snap = Snapshot(*self.__tuple_from_particle_data_row(row))
        snap.entry_type = EntryType(snap.entry_type)
        return snap

    def get_subfolder_name(self):
        return os.path.split(self.subfolder_path)[-1]

    def get_num_particles(self) -> int:
        return self.num_particles

    def event_entries(self) -> list[EventEntry]:
        for idx in range(len(self.event_dat)):
            yield self.get_event_entry_at(idx)

    def get_event_entry_at(self, idx) -> EventEntry:
        ev = self.event_dat[idx]
        entry_type = self.__id_to_entry(ev[0])
        if entry_type == EntryType.EVENT_ENTRY:
            return EventEntry(ev[1], entry_type, ev[2], ev[0], ev[3])
        else:
            return EventEntry(ev[1], entry_type, ev[2], -1, ev[3])

    def get_num_event_entries(self) -> int:
        return len(self.event_dat)

    def event_granular(self, i) -> list[Snapshot]:
        for j in range(len(self.particle_dats[i])):
            row = self.particle_dats[i][j]
            yield self.__snapshot_from_particle_data_row(row)

    def event_granular_raw(self, i) -> list[list[tuple]]:
        for j in range(len(self.particle_dats[i])):
            row = self.particle_dats[i][j]
            yield self.__tuple_from_particle_data_row(row)

    def __loop_window(self, i):
        for row in self.particle_dats[i]:
            if self.__id_to_entry(self.event_dat[int(row[0])][0]) == EntryType.WINDOW_CATCHUP and (
                int(row[0]) == len(self.event_dat) - 1 or
                self.__id_to_entry(self.event_dat[int(row[0] + 1)][0]) == EntryType.WINDOW_SUCCESS
            ): yield row

    def window_granular(self, i):
        for row in self.__loop_window(i):
            yield self.__snapshot_from_particle_data_row(row)

    def window_granular_raw(self, i) -> list[list[tuple]]:
        for row in self.__loop_window(i):
            yield self.__tuple_from_particle_data_row(row)


class Converter:
    """
    Convert raw data into a readable form using a reader.
    @param name_suffix: optional additional ifo about the converter's config as part of the subfolder name
    """
    def __init__(self, reader: DataReader, foldername: str, log_root_folder_path: str, name_suffix=""):
        self.reader = reader
        self.folder_name = foldername
        self.folder_path = os.path.join(log_root_folder_path, foldername)

        # Make folder name if not exists
        if foldername not in os.listdir(log_root_folder_path):
            os.mkdir(self.folder_path)

        # Find next available subfolder name and make it
        base_name = reader.get_subfolder_name() + name_suffix
        idx = 0
        cur_name = base_name + "-" + str(idx)
        while True:
            if cur_name in os.listdir(self.folder_path):
                idx += 1
                cur_name = base_name + "-" + str(idx)
            else: break
        self.subfolder_name = cur_name
        self.subfolder_path = os.path.join(self.folder_path, self.subfolder_name)
        os.mkdir(self.subfolder_path)

    @abstractmethod
    def convert(self):
        """Read the reader and generate output of the data in the format of this converter"""


class TextConverter(Converter):
    def __init__(self, reader: DataReader, foldername: str, log_root_folder_path: str,
                 ignore_zero_force_events=False, ignore_rollbacks=False,
                 is_binary=False, show_percentages=False):
        super().__init__(reader, foldername, log_root_folder_path, ("" if ignore_zero_force_events else "z") + ("" if ignore_rollbacks else "r") + ("b" if is_binary else ""))
        self.ignore_zero = ignore_zero_force_events
        self.ignore_rollbacks = ignore_rollbacks
        self.is_binary = is_binary
        self.show_percentages = show_percentages

    def convert(self):
        def convert_binary(x):
            return bin(struct.unpack('!i', struct.pack('!f', x))[0])

        def convert_identity(x):
            return x

        f = convert_binary if self.is_binary else convert_identity

        with open(os.path.join(self.subfolder_path, self.reader.get_subfolder_name() + "-event-entries"), "w") as file:
            cached_str = ""

            # Log event file
            for idx, event_entry in enumerate(self.reader.event_entries()):
                if event_entry.entry_type == EntryType.ROLLBACK:
                    # Dump data to file if rollbacks not ignored
                    if not self.ignore_rollbacks:
                        cached_str += "ROLLBACK\n"
                        file.write(cached_str)
                    cached_str = ""

                elif event_entry.entry_type == EntryType.WINDOW_SUCCESS:
                    cached_str += "-----------------\n"
                    file.write(cached_str)
                    cached_str = ""

                else:
                    cached_str += f"w={f(event_entry.window_idx)}  t={f(event_entry.t)}  " \
                                  f"ev={f(event_entry.event_identifier)}  Enet={f(event_entry.total_energy)}\n"

        # Log particle files
        for p in range(self.reader.get_num_particles()):
            with open(os.path.join(self.subfolder_path, self.reader.get_subfolder_name() + f"-P{p}"), "w") as file:

                cached_str = ""
                vel_at_window_start = None
                prev_vel = None
                events = self.reader.event_granular_raw(p)

                for idx, tup in enumerate(events):
                    if self.show_percentages: print(f"---- {idx}")
                    entry_type = tup[8]
                    event_entry_idx = int(tup[10])

                    if prev_vel is None or (not self.ignore_zero) or (prev_vel != tup[1]):
                        cached_str += f"w={f(tup[9])}  t={f(tup[2])}  ev={f(tup[7])}  " \
                                      f"x={f(tup[0])}  v={f(tup[1])}  E={f(tup[3])}  " \
                                      f"Ek={f(tup[4])}  Eg={f(tup[5])}  Ep={f(tup[6])}\n"
                    prev_vel = tup[1]

                    # If this is the last entry, flush the cache
                    if event_entry_idx >= self.reader.get_num_event_entries() - 1:
                        file.write(cached_str)

                    # Every particle is recorded at end of window. Use this to capture rollbacks and window ends
                    elif entry_type == EntryType.WINDOW_CATCHUP.value:

                        if self.reader.get_event_entry_at(event_entry_idx + 1).entry_type == EntryType.ROLLBACK:
                            if not self.ignore_rollbacks:
                                cached_str += "ROLLBACK\n"
                                file.write(cached_str)
                            prev_vel = vel_at_window_start

                        elif self.reader.get_event_entry_at(event_entry_idx + 1).entry_type == EntryType.WINDOW_SUCCESS:
                            cached_str += "-----------------\n"
                            file.write(cached_str)
                            vel_at_window_start = tup[1]

                        cached_str = ""


class MatplotlibConverter(Converter):
    def __init__(self, reader: DataReader, foldername: str, log_root_folder_path: str, only_total_energy=False):
        super().__init__(reader, foldername, log_root_folder_path, "-e" if only_total_energy else "")
        self.only_total_energy = only_total_energy

    def convert(self):
        if not self.only_total_energy:
            for p in range(self.reader.get_num_particles()):
                plt.plot([snap.t for snap in self.reader.window_granular(p)],
                         [snap.x for snap in self.reader.window_granular(p)])
            plt.xlabel("Time")
            plt.ylabel("Particle Height")
            plt.savefig(os.path.join(self.subfolder_path, f"{self.reader.get_subfolder_name()}-position.png"))
            plt.clf()

            for p in range(self.reader.get_num_particles()):
                plt.plot([snap.t for snap in self.reader.window_granular(p)],
                         [snap.v for snap in self.reader.window_granular(p)])
            plt.xlabel("Time")
            plt.ylabel("Particle Velocity")
            plt.savefig(os.path.join(self.subfolder_path, f"{self.reader.get_subfolder_name()}-velocity.png"))
            plt.clf()

            for p in range(self.reader.get_num_particles()):
                plt.plot([snap.t for snap in self.reader.window_granular(p)],
                         [snap.energy for snap in self.reader.window_granular(p)])
            plt.xlabel("Time")
            plt.ylabel("Particle Energy")
            plt.savefig(os.path.join(self.subfolder_path, f"{self.reader.get_subfolder_name()}-energy.png"))
            plt.clf()

            for p in range(self.reader.get_num_particles()):
                plt.plot([snap.t for snap in self.reader.window_granular(p)],
                         [snap.kinetic_energy for snap in self.reader.window_granular(p)])
            plt.xlabel("Time")
            plt.ylabel("Particle Kinetic Energy")
            plt.savefig(os.path.join(self.subfolder_path, f"{self.reader.get_subfolder_name()}-kineticenergy.png"))
            plt.clf()

            for p in range(self.reader.get_num_particles()):
                plt.plot([snap.t for snap in self.reader.window_granular(p)],
                         [snap.potential_energy for snap in self.reader.window_granular(p)])
            plt.xlabel("Time")
            plt.ylabel("Particle Potential Energy")
            plt.savefig(os.path.join(self.subfolder_path, f"{self.reader.get_subfolder_name()}-potentialenergy.png"))
            plt.clf()

            for p in range(self.reader.get_num_particles()):
                plt.plot([snap.t for snap in self.reader.window_granular(p)],
                         [snap.penalty_energy for snap in self.reader.window_granular(p)])
            plt.xlabel("Time")
            plt.ylabel("Particle Penalty Energy")
            plt.savefig(os.path.join(self.subfolder_path, f"{self.reader.get_subfolder_name()}-penaltyenergy.png"))
            plt.clf()

        plt.plot([entry.t for entry in self.reader.event_entries() if entry.entry_type.value < 2],
                 [entry.total_energy for entry in self.reader.event_entries() if entry.entry_type.value < 2])
        plt.xlabel("Time")
        plt.ylabel("Total Energy")
        plt.savefig(os.path.join(self.subfolder_path, f"{self.reader.get_subfolder_name()}-totalenergy.png"))
        plt.clf()


class TensorboardConverter(Converter):
    """
    To install tensorboard, simple run `pip install tensorboard`
    Then run tensorboard command in terminal with `--logdir <folder path>`
    Add `--samples_per_plugin scalars=999999999` to ensure datapoints aren't skipped
    """
    def __init__(self, reader: DataReader, foldername: str, log_root_folder_path: str):
        super().__init__(reader, foldername, log_root_folder_path)

    def convert(self):
        writer = SummaryWriter(os.path.join(self.subfolder_path, self.reader.get_subfolder_name()))

        for p in range(self.reader.get_num_particles()):
            for snap in self.reader.window_granular(p):
                writer.add_scalar(f"PosP{p}", snap.x, int(snap.t * 100000))
                writer.add_scalar(f"VelP{p}", snap.v, int(snap.t * 100000))
                writer.add_scalar(f"EnergyP{p}", snap.energy, int(snap.t * 100000))
                writer.add_scalar(f"KineticEnergyP{p}", snap.kinetic_energy, int(snap.t * 100000))
                writer.add_scalar(f"PotentialEnergyP{p}", snap.potential_energy, int(snap.t * 100000))
                writer.add_scalar(f"PenaltyEnergyP{p}", snap.penalty_energy, int(snap.t * 100000))

        for entry in self.reader.event_entries():
            if entry.entry_type.value < 2:
                writer.add_scalar("TotalEnergy", entry.total_energy, int(entry.t * 100000))

        writer.close()

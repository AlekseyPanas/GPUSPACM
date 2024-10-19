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


@dataclass(unsafe_hash=True)
class Snapshot:
    """If snap_type is ROLLBACK then rest of data is -1"""
    x: float
    v: float
    t: float
    energy: Optional[float]
    kinetic_energy: Optional[float]
    potential_energy: Optional[float]
    penalty_energy: Optional[float]
    event_identifier: int


class DataReader:
    """Reads a log file of a specific type and parses it"""
    @abstractmethod
    def get_file_name(self) -> str:
        """Get the name of the file being parsed (not the whole path, just the file name)"""

    @abstractmethod
    def get_num_particles(self) -> int:
        """Get the number of particles in the simulation that generated the data file being read by this reader"""

    @abstractmethod
    def event_entries(self) -> list[EventEntry]:
        """Return an iterable of all event entries in the simulation"""

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

        self.num_particles: int = len(os.listdir(self.subfolder_path)) - 1

        self.event_filepath = None
        self.particle_filepaths: list[str | None] = [None] * self.num_particles

        for filename in os.listdir(self.subfolder_path):
            if filename.endswith("-events.npy"):
                self.event_filepath = os.path.join(self.subfolder_path, filename)
            else:
                self.particle_filepaths[int(filename[-5])] = filename

        assert self.event_filepath is not None

        self.event_dat = np.load(self.event_filepath, mmap_mode="r")
        self.particle_dats = [np.load(path, mmap_mode="r") for path in self.particle_filepaths]

    def get_file_name(self):
        return os.path.split(self.subfolder_path)[-1]

    def get_num_particles(self) -> int:
        return self.num_particles

    def event_entries(self) -> list[EventEntry]:
        for ev in self.event_dat:
            entry_type = {-3: EntryType.WINDOW_SUCCESS, -2: EntryType.ROLLBACK, -1: EntryType.WINDOW_CATCHUP}.get(ev[0], EntryType.EVENT_ENTRY)
            if entry_type == EntryType.EVENT_ENTRY:
                yield EventEntry(ev[1], entry_type, ev[2], -1)
            else:
                yield EventEntry(ev[1], entry_type, ev[2], ev[0])

    def event_granular(self, i) -> list[Snapshot]:
        for j in range(len(self.particle_dats[i])):
            row = self.particle_dats[i][j]
            yield Snapshot(row[1], row[2], self.event_dat[row[0]][1], row[3], row[4], row[5], row[6], self.event_dat[row[0]][0])

    def event_granular_raw(self, i) -> list[list[tuple]]:
        for j in range(len(self.particle_dats[i])):
            row = self.particle_dats[i][j]
            yield row[1], row[2], self.event_dat[row[0]][1], row[3], row[4], row[5], row[6], self.event_dat[row[0]][0]

    def __loop_window(self, i):
        for row in self.particle_dats[i]:
            if self.event_dat[row[0]] == 

            if self.event_dat[r][0][7] == EntryType.WINDOW_CATCHUP.value and (
                    self.event_dat.shape[0] == r + 1 or self.event_dat[r + 1][0][7] != EntryType.ROLLBACK.value
            ): yield r

    def window_granular(self, i):
        for r in self.__loop_window(i):
            yield [Snapshot(particle[0], particle[1], particle[2], particle[3],
                            particle[4], particle[5], particle[6], EntryType(particle[7]), particle[8])
                   for particle in self.event_dat[r]]

    def window_granular_raw(self, i) -> list[list[tuple]]:
        for r in self.__loop_window(i):
            yield [(particle[0], particle[1], particle[2], particle[3],
                            particle[4], particle[5], particle[6], particle[7], particle[8])
                   for particle in self.event_dat[r]]


class Converter:
    def __init__(self, reader: DataReader, foldername: str, log_root_folder_path: str):
        self.reader = reader
        self.folder_name = foldername
        self.folder_path = os.path.join(log_root_folder_path, foldername)

        if foldername not in os.listdir(log_root_folder_path):
            os.mkdir(self.folder_path)

        self.output_number = self.get_latest_output_number() + 1

    def get_latest_output_number(self):
        vals = [self.parse_last_part(f.split("-")[-1]) for f in os.listdir(self.folder_path) if
                "-".join(f.split("-")[:-self.number_of_suffixes()]) == self.reader.get_file_name()]
        return -1 if len(vals) == 0 else max(vals)

    @abstractmethod
    def number_of_suffixes(self) -> int:
        """Filenames of converted files are saved as <name of npy file>-suffix1-suffix2-suffix3-...-suffixn.
        Depending on the converter, there may be varying number of suffixes which affects how the
        latest output number is computed. This should return the number of suffixes in the filenames of the
        inheriting converter"""

    @abstractmethod
    def parse_last_part(self, suffix: str) -> int:
        """Parses the final suffix which contains the output number. Some files may have extensions which need removal
        in which case this method needs to implement the parsing logic. e.g 0.png -> 0"""

    @abstractmethod
    def convert(self):
        """Read the reader and generate output of the data in the format of this converter"""


class TextConverter(Converter):
    def __init__(self, reader: DataReader, foldername: str, log_root_folder_path: str,
                 ignore_zero_force_events=False, ignore_rollbacks=False,
                 is_binary=False, show_percentages=False):
        super().__init__(reader, foldername, log_root_folder_path)
        self.ignore_zero = ignore_zero_force_events
        self.ignore_rollbacks = ignore_rollbacks
        self.is_binary = is_binary
        self.show_percentages = show_percentages

    def number_of_suffixes(self) -> int: return 2

    def parse_last_part(self, suffix: str) -> int: return int(suffix)

    def convert(self):
        def convert_binary(x):
            return bin(struct.unpack('!i', struct.pack('!f', x))[0])

        def convert_identity(x):
            return x

        for p in range(self.reader.get_num_particles()):
            with open(os.path.join(self.folder_path,
                                   self.reader.get_file_name() + f"-P{p}-{self.output_number}"), "w") as file:
                f = convert_binary if self.is_binary else convert_identity
                cached_str = ""
                just_saw_end_of_window = False
                prev_vel = -5
                events = self.reader.event_granular_raw()
                for idx, tups in enumerate(events):
                    if self.show_percentages: print(f"---- {idx}")
                    snapshot_type = tups[p][7]

                    if snapshot_type == EntryType.ROLLBACK.value:
                        # Dump data to file if rollbacks not ignored
                        if not self.ignore_rollbacks:
                            cached_str += "ROLLBACK\n"
                            file.write(cached_str)
                        cached_str = ""
                        just_saw_end_of_window = False

                    else:
                        if just_saw_end_of_window:
                            cached_str += "-----------------\n"
                            file.write(cached_str)
                            cached_str = ""
                            just_saw_end_of_window = False

                        if (not self.ignore_zero) or tups[p][1] != prev_vel or \
                                snapshot_type == EntryType.WINDOW_CATCHUP.value:
                            cached_str += f"t={f(tups[p][2])}  ev={f(tups[p][8])}  " \
                                          f"x={f(tups[p][0])}  v={f(tups[p][1])}  " \
                                          f"E={f(tups[p][3])}  Ek={f(tups[p][4])}  " \
                                          f"Eg={f(tups[p][5])}  Ep={f(tups[p][6])}\n"
                        prev_vel = tups[p][1]

                    if snapshot_type == EntryType.WINDOW_CATCHUP.value:
                        just_saw_end_of_window = True


class MatplotlibConverter(Converter):
    def number_of_suffixes(self) -> int: return 2

    def parse_last_part(self, suffix: str) -> int: return int(suffix.split(".")[0])

    def convert(self):
        subfolder_path = os.path.join(self.folder_path, f"{self.reader.get_file_name()}-{self.output_number}")
        os.mkdir(subfolder_path)

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].x for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Height")
        plt.savefig(os.path.join(subfolder_path, f"{self.reader.get_file_name()}-position-{self.output_number}.png"))
        plt.clf()

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].v for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Velocity")
        plt.savefig(os.path.join(subfolder_path, f"{self.reader.get_file_name()}-velocity-{self.output_number}.png"))
        plt.clf()

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].energy for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Energy")
        plt.savefig(os.path.join(subfolder_path, f"{self.reader.get_file_name()}-energy-{self.output_number}.png"))
        plt.clf()

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].kinetic_energy for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Kinetic Energy")
        plt.savefig(os.path.join(subfolder_path, f"{self.reader.get_file_name()}-kineticenergy-{self.output_number}.png"))
        plt.clf()

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].potential_energy for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Potential Energy")
        plt.savefig(os.path.join(subfolder_path, f"{self.reader.get_file_name()}-potentialenergy-{self.output_number}.png"))
        plt.clf()

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].penalty_energy for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Penalty Energy")
        plt.savefig(os.path.join(subfolder_path, f"{self.reader.get_file_name()}-penaltyenergy-{self.output_number}.png"))
        plt.clf()


class TensorboardConverter(Converter):
    def __init__(self, reader: DataReader, foldername: str, log_root_folder_path: str, only_window: bool):
        super().__init__(reader, foldername, log_root_folder_path)
        self.only_window = only_window  # When true only records snapshots at the end of each window, ignores rollback

    def number_of_suffixes(self) -> int: return 1

    def parse_last_part(self, suffix: str) -> int: return int(suffix)

    def convert(self):
        writer = SummaryWriter(os.path.join(self.folder_path, self.reader.get_file_name() + f"-{self.output_number}"))

        if self.only_window:
            self.positions = [[snap.x for snap in snaps] for snaps in self.reader.window_granular()]
            self.velocities = [[snap.v for snap in snaps] for snaps in self.reader.window_granular()]
            self.energies = [[snap.energy for snap in snaps] for snaps in self.reader.window_granular()]
            self.kinetic_energies = [[snap.kinetic_energy for snap in snaps] for snaps in self.reader.window_granular()]
            self.potential_energies = [[snap.potential_energy for snap in snaps] for snaps in self.reader.window_granular()]
            self.penalty_energies = [[snap.penalty_energy for snap in snaps] for snaps in self.reader.window_granular()]
            self.times = [snaps[0].t for snaps in self.reader.window_granular()]

            # All the lists should be the same length so choosing self.positions is arbitrary
            for i in range(len(self.positions)):
                for p in range(len(self.positions[i])):
                    writer.add_scalar(f"PosP{p}", self.positions[i][p], int(self.times[i] * 100000))
                    writer.add_scalar(f"VelP{p}", self.velocities[i][p], int(self.times[i] * 100000))
                    writer.add_scalar(f"EnergyP{p}", self.energies[i][p], int(self.times[i] * 100000))
                    writer.add_scalar(f"KineticEnergyP{p}", self.kinetic_energies[i][p], int(self.times[i] * 100000))
                    writer.add_scalar(f"PotentialEnergyP{p}", self.potential_energies[i][p], int(self.times[i] * 100000))
                    writer.add_scalar(f"PenaltyEnergyP{p}", self.penalty_energies[i][p], int(self.times[i] * 100000))

        else:
            self.positions = [[snap.x for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != EntryType.ROLLBACK]
            self.velocities = [[snap.v for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != EntryType.ROLLBACK]
            self.energies = [[snap.energy for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != EntryType.ROLLBACK]
            self.kinetic_energies = [[snap.kinetic_energy for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != EntryType.ROLLBACK]
            self.potential_energies = [[snap.potential_energy for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != EntryType.ROLLBACK]
            self.penalty_energies = [[snap.penalty_energy for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != EntryType.ROLLBACK]
            self.times = [snaps[0].t for snaps in self.reader.event_granular() if snaps[0].snap_type != EntryType.ROLLBACK]
            self.rollback_numbers = []
            self.cur_rollback = 0
            for snaps in self.reader.event_granular():
                if snaps[0].snap_type == EntryType.ROLLBACK:
                    self.cur_rollback += 1
                else:
                    self.rollback_numbers.append(self.cur_rollback)
            assert len(self.rollback_numbers) == len(self.times)

            # All the lists should be the same length so choosing self.positions is arbitrary
            for i in range(len(self.positions)):
                for p in range(len(self.positions[i])):
                    writer.add_scalar(f"PosP{p}/{self.rollback_numbers[i]}", self.positions[i][p], int(self.times[i] * 100000))
                    writer.add_scalar(f"VelP{p}/{self.rollback_numbers[i]}", self.velocities[i][p], int(self.times[i] * 100000))
                    writer.add_scalar(f"EnergyP{p}/{self.rollback_numbers[i]}", self.energies[i][p], int(self.times[i] * 100000))
                    writer.add_scalar(f"KineticEnergyP{p}/{self.rollback_numbers[i]}", self.kinetic_energies[i][p], int(self.times[i] * 100000))
                    writer.add_scalar(f"PotentialEnergyP{p}/{self.rollback_numbers[i]}", self.potential_energies[i][p], int(self.times[i] * 100000))
                    writer.add_scalar(f"PenaltyEnergyP{p}/{self.rollback_numbers[i]}", self.penalty_energies[i][p], int(self.times[i] * 100000))

            layout = {
                f"SimulationP{p}": {
                    f"PosP{p}": ["Multiline", [f"PosP{p}/{i}" for i in range(self.cur_rollback + 1)]],
                    f"VelP{p}": ["Multiline", [f"VelP{p}/{i}" for i in range(self.cur_rollback + 1)]],
                    f"EnergyP{p}": ["Multiline", [f"EnergyP{p}/{i}" for i in range(self.cur_rollback + 1)]],
                    f"KineticEnergyP{p}": ["Multiline", [f"KineticEnergyP{p}/{i}" for i in range(self.cur_rollback + 1)]],
                    f"PotentialEnergyP{p}": ["Multiline", [f"PotentialEnergyP{p}/{i}" for i in range(self.cur_rollback + 1)]],
                    f"PenaltyEnergyP{p}": ["Multiline", [f"PenaltyEnergyP{p}/{i}" for i in range(self.cur_rollback + 1)]]
                } for p in range(self.reader.get_num_particles())
            }

            writer.add_custom_scalars(layout)
        writer.close()

        self.output_number += 1

"""
Simulations save data in npy files. This script provides utilities to read and output that data into
other formats such as txt, tensorboard, or matplotlib
"""
from tensorboardX import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Iterable
from enum import IntEnum
import os
import struct


class SnapshotType(IntEnum):
    EVENT_GRANULAR = 0
    WINDOW_GRANULAR = 1
    ROLLBACK = 2


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
    snap_type: SnapshotType
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
    def event_granular(self) -> list[list[Snapshot]]:
        """Return an iterable of all recorded snapshots of the simulation for all particles"""

    @abstractmethod
    def event_granular_raw(self) -> list[list[tuple]]:
        """Return an iterable of all recorded snapshots of the simulation for all particles. The data returned
        is a raw tuple containing unparsed numerical data corresponding to parameters in the Snapshot class"""

    @abstractmethod
    def window_granular(self) -> list[list[Snapshot]]:
        """Return iterable of snapshots at the end of successful (non-rollback) time windows
        for all particles throughout the simulation"""

    @abstractmethod
    def window_granular_raw(self) -> list[list[tuple]]:
        """Return iterable of snapshots at the end of successful (non-rollback) time windows
        for all particles throughout the simulation. The data returned is a raw tuple containing
        unparsed numerical data corresponding to parameters in the Snapshot class"""


class NumpyDataReader(DataReader):
    """Parses an .npy file with snapshot data. The snapshot data is expected as a 3D array
    where each 'row' is a list of snapshots for each particle. A snapshot is stored as an array
    of values in the order of the Snapshot dataclass"""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dat = np.load(filepath, mmap_mode="r")

    def get_file_name(self):
        return os.path.split(self.filepath)[-1]

    def get_num_particles(self) -> int:
        return self.dat.shape[1]

    def event_granular(self):
        for row in self.dat:
            yield [Snapshot(particle[0], particle[1], particle[2], particle[3],
                            particle[4], particle[5], particle[6], SnapshotType(int(particle[7])),
                            particle[8]) for particle in row]

    def __loop_window(self):
        for r in range(self.dat.shape[0]):
            if self.dat[r][0][7] == SnapshotType.WINDOW_GRANULAR.value and (
                    self.dat.shape[0] == r + 1 or self.dat[r + 1][0][7] != SnapshotType.ROLLBACK.value
            ): yield r

    def window_granular(self):
        for r in self.__loop_window():
            yield [Snapshot(particle[0], particle[1], particle[2], particle[3],
                            particle[4], particle[5], particle[6], SnapshotType(particle[7]), particle[8])
                   for particle in self.dat[r]]

    def event_granular_raw(self) -> list[list[tuple]]:
        for row in self.dat:
            yield [(particle[0], particle[1], particle[2], particle[3],
                    particle[4], particle[5], particle[6], particle[7],
                    particle[8]) for particle in row]

    def window_granular_raw(self) -> list[list[tuple]]:
        for r in self.__loop_window():
            yield [(particle[0], particle[1], particle[2], particle[3],
                            particle[4], particle[5], particle[6], particle[7], particle[8])
                   for particle in self.dat[r]]


class Converter:
    def __init__(self, reader: DataReader, foldername: str):
        self.reader = reader
        self.folder_name = foldername

        if foldername not in os.listdir("."):
            os.mkdir(f"./{foldername}")

        self.output_number = self.get_latest_output_number() + 1

    def get_latest_output_number(self):
        vals = [self.parse_last_part(f.split("-")[-1]) for f in os.listdir(f"./{self.folder_name}") if
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
    def __init__(self, reader: DataReader, foldername: str, ignore_zero_force_events=False, ignore_rollbacks=False,
                 is_binary=False, show_percentages=False):
        super().__init__(reader, foldername)
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
            with open(os.path.join(".", self.folder_name,
                                   self.reader.get_file_name() + f"-P{p}-{self.output_number}"), "w") as file:
                f = convert_binary if self.is_binary else convert_identity
                cached_str = ""
                just_saw_end_of_window = False
                prev_vel = -5
                events = self.reader.event_granular_raw() if self.is_binary else self.reader.event_granular()
                for idx, tups in enumerate(events):
                    if self.show_percentages: print(f"---- {idx / len(events)}%")
                    snapshot_type = tups[p][7] if self.is_binary else tups[p][7].value

                    if snapshot_type == SnapshotType.ROLLBACK.value:
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
                                snapshot_type == SnapshotType.WINDOW_GRANULAR.value:
                            cached_str += f"t={f(tups[p][2])}  ev={f(tups[p][8])}  " \
                                          f"x={f(tups[p][0])}  v={f(tups[p][1])}  " \
                                          f"E={f(tups[p][3])}  Ek={f(tups[p][4])}  " \
                                          f"Eg={f(tups[p][5])}  Ep={f(tups[p][6])}\n"
                        prev_vel = tups[p][1]

                    if snapshot_type == SnapshotType.WINDOW_GRANULAR.value:
                        just_saw_end_of_window = True


class MatplotlibConverter(Converter):
    def number_of_suffixes(self) -> int: return 2

    def parse_last_part(self, suffix: str) -> int: return int(suffix.split(".")[0])

    def convert(self):
        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].x for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Height")
        plt.savefig(os.path.join(self.folder_name, f"{self.reader.get_file_name()}-position-{self.output_number}.png"))
        plt.clf()

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].v for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Velocity")
        plt.savefig(os.path.join(self.folder_name, f"{self.reader.get_file_name()}-velocity-{self.output_number}.png"))
        plt.clf()

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].energy for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Energy")
        plt.savefig(os.path.join(self.folder_name, f"{self.reader.get_file_name()}-energy-{self.output_number}.png"))
        plt.clf()

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].kinetic_energy for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Kinetic Energy")
        plt.savefig(os.path.join(self.folder_name, f"{self.reader.get_file_name()}-kineticenergy-{self.output_number}.png"))
        plt.clf()

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].potential_energy for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Potential Energy")
        plt.savefig(os.path.join(self.folder_name, f"{self.reader.get_file_name()}-potentialenergy-{self.output_number}.png"))
        plt.clf()

        for p in range(self.reader.get_num_particles()):
            plt.plot([snaps[p].t for snaps in self.reader.window_granular()],
                     [snaps[p].penalty_energy for snaps in self.reader.window_granular()])
        plt.xlabel("Time")
        plt.ylabel("Particle Penalty Energy")
        plt.savefig(os.path.join(self.folder_name, f"{self.reader.get_file_name()}-penaltyenergy-{self.output_number}.png"))
        plt.clf()


class TensorboardLogger(Converter):
    def __init__(self, reader: DataReader, foldername: str, only_window: bool):
        super().__init__(reader, foldername)
        self.only_window = only_window  # When true only records snapshots at the end of each window, ignores rollback

    def number_of_suffixes(self) -> int: return 1

    def parse_last_part(self, suffix: str) -> int: return int(suffix)

    def convert(self):
        writer = SummaryWriter(os.path.join(self.folder_name, self.reader.get_file_name() + f"-{self.output_number}"))

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
            self.positions = [[snap.x for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != SnapshotType.ROLLBACK]
            self.velocities = [[snap.v for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != SnapshotType.ROLLBACK]
            self.energies = [[snap.energy for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != SnapshotType.ROLLBACK]
            self.kinetic_energies = [[snap.kinetic_energy for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != SnapshotType.ROLLBACK]
            self.potential_energies = [[snap.potential_energy for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != SnapshotType.ROLLBACK]
            self.penalty_energies = [[snap.penalty_energy for snap in snaps] for snaps in self.reader.event_granular() if snaps[0].snap_type != SnapshotType.ROLLBACK]
            self.times = [snaps[0].t for snaps in self.reader.event_granular() if snaps[0].snap_type != SnapshotType.ROLLBACK]
            self.rollback_numbers = []
            self.cur_rollback = 0
            for snaps in self.reader.event_granular():
                if snaps[0].snap_type == SnapshotType.ROLLBACK:
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


def choose_option_from_list(opts: list[str]) -> int:
    print("\n".join(f"[{idx}] {opt}" for idx, opt in enumerate(opts)))
    print("=============")
    while True:
        try:
            opt_idx = int(input("Select an option: ").strip())
            if opt_idx >= len(opts) or opt_idx < 0:
                print("Bruh that ain't one of the given options dawg")
            else:
                break
        except Exception:
            print("Bruh that ain't a number dawg")
    return opt_idx


if __name__ == "__main__":
    if "npdat" not in os.listdir("."):
        os.mkdir("./npdat")

    file_idx = choose_option_from_list(os.listdir("./npdat"))

    filepath = os.path.join(".", "npdat", os.listdir("./npdat")[file_idx])
    reader = NumpyDataReader(filepath)

    options = ["Text", "Matplotlib", "Tensorboard", "Tensorboard without Rollback"]
    opt_idx = choose_option_from_list(options)

    if opt_idx == 0:
        TextConverter(reader, "textlogs", False, False, True, True).convert()
    elif opt_idx == 1:
        MatplotlibConverter(reader, "matplotlibplots").convert()
    elif opt_idx == 2:
        TensorboardLogger(reader, "runs", False).convert()
    elif opt_idx == 3:
        TensorboardLogger(reader, "runs", True).convert()
    else:
        print("Sus.... this codepath should be impossible")

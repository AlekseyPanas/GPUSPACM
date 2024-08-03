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
    def window_granular(self) -> list[list[Snapshot]]:
        """Return iterable of snapshots at the end of successful (non-rollback) time windows
        for all particles throughout the simulation"""


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
                            particle[4], particle[5], particle[6], SnapshotType(int(particle[7]))) for particle in row]

    def window_granular(self):
        for r in range(self.dat.shape[0]):
            if self.dat[r][0][7] == SnapshotType.WINDOW_GRANULAR.value and (
                    self.dat.shape[0] == r + 1 or self.dat[r + 1][0][7] != SnapshotType.ROLLBACK.value
            ):
                yield [Snapshot(particle[0], particle[1], particle[2], particle[3],
                                particle[4], particle[5], particle[6], SnapshotType(particle[7]))
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
    def number_of_suffixes(self) -> int: return 2

    def parse_last_part(self, suffix: str) -> int: return int(suffix)

    def convert(self):
        for p in range(self.reader.get_num_particles()):
            with open(os.path.join(".", self.folder_name,
                                   self.reader.get_file_name() + f"-P{p}-{self.output_number}"), "w") as file:
                for snapshots in self.reader.event_granular():
                    if snapshots[p].snap_type == SnapshotType.ROLLBACK:
                        file.write("ROLLBACK\n")
                    else:
                        file.write(f"t={snapshots[p].t}  x={snapshots[p].x}  v={snapshots[p].v}  "
                                   f"E={snapshots[p].energy}  Ek={snapshots[p].kinetic_energy}  "
                                   f"Eg={snapshots[p].potential_energy}  Ep={snapshots[p].penalty_energy}\n")


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

    def number_of_suffixes(self) -> int: return 2

    def parse_last_part(self, suffix: str) -> int: return int(suffix)

    def convert(self):
        writer = SummaryWriter(self.folder_name + f"-{self.output_number}")

        # TODO: Fix this method, currently pasted from old code

        if self.only_window:
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
                } for p in range(self.num_particles)
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
        TextConverter(reader, "textlogs").convert()
    elif opt_idx == 1:
        MatplotlibConverter(reader, "matplotlibplots").convert()
    elif opt_idx == 2:
        pass
    elif opt_idx == 3:
        pass
    else:
        print("Sus.... this codepath should be impossible")

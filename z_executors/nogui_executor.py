from __future__ import annotations
from z_sims.core.data import YieldType, Snapshot
from z_sims.core.events import PenaltyEvent
from z_sims.core.sim import Sim
from z_sims.core.objects import Collidable
from enum import IntEnum
import time


class NoGuiExecutor:
    class RunModes(IntEnum):
        STM = 0,
        RTM = 1,
        EM = 2,
        RM = 3,
        WM = 4

    def __init__(self, sim: Sim):
        self.sim = sim
        self.mode = self.RunModes.STM

        # Sim Time Mode (STM): Run up to a certain time in the sim
        self.stm_time_to_run = 25

        # Real Time Mode (RTM): Run for a certain duration in real time
        self.rtm_time_to_run = 0

        # Event Mode (EM): Run a certain number of events
        self.em_num_events = 0
        self.em_accum_rollbacks = False

        # Rollback Mode (RM): Run a certain number of rollbacks
        self.rm_num_rollbacks = 0

        # Window Mode (WM): Run a certain number of windows
        self.wm_num_windows = 0

    def set_mode_sim_time(self, sim_time_to_run: float) -> NoGuiExecutor:
        self.stm_time_to_run = sim_time_to_run
        return self

    def set_mode_real_time(self, real_time_to_run: float) -> NoGuiExecutor:
        self.rtm_time_to_run = real_time_to_run
        return self

    def set_mode_event(self, num_events: int, accumulate_rollbacks: bool) -> NoGuiExecutor:
        self.em_num_events = num_events
        self.em_accum_rollbacks = accumulate_rollbacks
        return self

    def set_mode_rollback(self, num_rollbacks) -> NoGuiExecutor:
        self.rm_num_rollbacks = num_rollbacks
        return self

    def set_mode_window(self, num_windows) -> NoGuiExecutor:
        self.wm_num_windows = num_windows
        return self

    def run(self):
        s = self.sim.run_sim()
        start_time = time.time()
        event_count_total = 0
        event_count_window = 0
        window_count = 0
        rollback_count = 0

        print("Running sim...")
        while True:
            typ, _ = next(s)

            if self.mode == self.RunModes.STM:
                sim_time = self.sim.get_sim_time()
                print(f"Sim Time: {sim_time} ---- {self.stm_time_to_run}")
                if sim_time > self.stm_time_to_run:
                    break
            elif self.mode == self.RunModes.RTM:
                real_time = time.time() - start_time
                print(f"Real Time: {real_time} ---- {self.rtm_time_to_run}")
                if real_time > self.rtm_time_to_run:
                    break
            elif self.mode == self.RunModes.EM:
                if typ == YieldType.EVENT_FINISHED:
                    event_count_window += 1
                elif (self.em_accum_rollbacks and typ == YieldType.TIME_WINDOW_FINISHED_ROLLBACK) or \
                        typ == YieldType.TIME_WINDOW_FINISHED:
                    event_count_total += event_count_window
                    event_count_window = 0
                total_events_processed = event_count_total + event_count_window
                print(f"Events Processed: {total_events_processed} ---- {self.em_num_events}")
                if total_events_processed > self.em_num_events:
                    break
            elif self.mode == self.RunModes.RM:
                if typ == YieldType.TIME_WINDOW_FINISHED_ROLLBACK:
                    rollback_count += 1
                print(f"Rollbacks Processed: {rollback_count} ---- {self.rm_num_rollbacks}")
                if rollback_count > self.rm_num_rollbacks:
                    break
            elif self.mode == self.RunModes.WM:
                if typ == YieldType.TIME_WINDOW_FINISHED:
                    window_count += 1
                print(f"Windows Processed: {window_count} ---- {self.wm_num_windows}")
                if window_count > self.wm_num_windows:
                    break
        print("Running sim finished!")

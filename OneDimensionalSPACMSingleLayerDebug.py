from __future__ import  annotations

import numpy as np
import pygame
from matplotlib import pyplot as plt
import math
import numpy
from dataclasses import dataclass
from typing import Optional
import heapq
from abc import abstractmethod
from enum import IntEnum
import random
import time
import colorsys
from tensorboardX import SummaryWriter
from datetime import datetime
pygame.init()


@dataclass(unsafe_hash=True)
class Snapshot:
    x: float
    v: float
    t: float
    energy: Optional[float]
    has_velocity_changed: bool


def get_next_time(start_time: float, dt: float, cur_time: float):
    """
    :param start_time: time when this clock started ticking
    :param dt: timestep of globally ticking clock
    :param cur_time: the current timestamp
    :return: the time of the next tick of a global clock that ticked from start_time at intervals of dt. for example,
    get_next_time(0, 3, 7) returns 9 because the global clock ticks at 3, 6, 9, ..., and 9 is the next tick after 7.
    the returned timestamp excludes cur_time, so get_next_time(0, 2, 4) returns 6 and not 4
    """
    return start_time + ((((cur_time - start_time) // dt) + 1) * dt)


class Event:
    """ Recurring event applying force to all particles in the simulation """
    def __init__(self, timestep: float):
        self.h = timestep

    @abstractmethod
    def get_force(self, target: Particle):
        """Return the force, in newtons, that this even applies on the given particle"""

    def __lt__(self, other):
        """Completely arbitrary comparison so that code doesn't crash"""
        return self.h < other.h


class GravityEvent(Event):
    """ Force of Gravity """
    def __init__(self, timestep: float, accel_gravity: float):
        super().__init__(timestep)
        self.g = accel_gravity

    def get_force(self, target: Particle) -> float:
        return target.mass * self.g


class InfinitePenaltyEvent(Event):
    """Infinitely large penalty layer active below some threshold"""
    def __init__(self, thresh: float, timestep: float, stiffness: float):
        super().__init__(timestep)
        self.thresh = thresh
        self.stiffness = stiffness

    def get_gap(self, q: float):
        return q - self.thresh

    def get_force(self, target: Particle):
        gap = target.get_latest_position() - self.thresh
        if gap <= 0:
            return self.stiffness * abs(gap)
        else:
            return 0

    def get_stiffness(self):
        return self.stiffness


class PenaltyEvent(Event):
    """ Single collision penalty layer for a collideable object """
    def __init__(self, base_timestep: float, owner: Collidable, layer: int):
        super().__init__(owner.get_lth_timestep(layer, base_timestep))
        self.owner = owner
        self.layer = layer

    def get_gap(self, q: float):
        """Return gap function g(q) for a particle position q"""
        return abs(self.owner.get_latest_position() - q) - self.owner.get_lth_thickness(self.layer)

    def get_force(self, target: Particle) -> float:
        if target is self.owner: return 0

        # the latest position is guaranteed to be at an equal timestep because SPACM sim code always updates
        # positions of EVERY particle before computing forces. If stencils are ever implemented this may no
        # longer be the case
        targ_x = target.get_latest_position()
        own_x = self.owner.get_latest_position()

        gap = self.get_gap(targ_x)
        if gap > 0:
            return 0
        else:
            return self.owner.get_lth_stiffness(self.layer) * gap * (-1 if targ_x > own_x else 1)


class Collidable:
    next_id = 0

    def __init__(self, collision_stiffness: float, first_layer_thickness: float, x: float):
        self.r1 = collision_stiffness  # first penalty layer's collision stiffness
        self.n1 = first_layer_thickness
        self.init_x = x
        self.active_penalties: list[PenaltyEvent] = []
        self.obj_id = Collidable.next_id
        Collidable.next_id += 1

    def get_lth_thickness(self, l: int) -> float:
        return self.n1 * (l ** (-0.25))  # ACM paper section 4

    def get_lth_stiffness(self, l: int) -> float:
        return self.r1 * (l ** 3)  # ACM paper section 4

    def get_lth_timestep(self, l: int, base_dt: float, fudge=1e-4) -> float:
        return base_dt / l / math.sqrt(l + fudge)

    def get_velocity_changed_timestamps(self, window_start: float, window_end: float) -> set[float]:
        """
        When detecting collisions across a rollback window, we need the snapshots where velocity of the object
        changed to compute piecewise linear collision intervals
        :return: a list of velocity-changed snapshots including window_start and window_end by default
        """
        return {window_start, window_end}

    def get_latest_position(self) -> float:
        return self.init_x

    def get_pos_at_time(self, t):
        return self.init_x


class Particle(Collidable):
    """
    current_snapshots: snapshots in the current rollback window. First snapshot in this list is the start of
    this rollback window
    previous_snapshots: rollback windows which have passed.
    """
    def __init__(self, collision_stiffness: float, first_layer_thickness: float,
                 x: float, v: float, cur_time: float, mass: float):
        super().__init__(collision_stiffness, first_layer_thickness, x)

        self.previous_snapshots: list[Snapshot] = []
        self.current_snapshots: list[Snapshot] = []

        self.current_snapshots.append(Snapshot(x, v, cur_time, None, True))

        self.mass = mass

    def get_velocity_changed_timestamps(self, window_start: float, window_end: float) -> set[float]:
        return {self.current_snapshots[0].t, self.current_snapshots[-1].t}.union(
            {snap.t for snap in self.current_snapshots if snap.has_velocity_changed})

    def get_latest_position(self) -> float:
        return self.current_snapshots[-1].x

    def get_pos_at_time(self, t) -> float:
        # assumes snapshots are ordered temporally in their list which is a consequence of the code
        before = None
        for snap in self.current_snapshots:
            if snap.t == t:
                return snap.x
            elif snap.t < t:
                before = snap
            else:
                break
        return before.x + ((t - before.t) * before.v)


class YieldType(IntEnum):
    EVENT_FINISHED = 0
    TIME_WINDOW_FINISHED = 1
    TIME_WINDOW_FINISHED_ROLLBACK = 2


class Logger:
    @abstractmethod
    def rollback(self):
        """Notifies the logger that a rollback has occurred. Subsequent datapoints will start in overlap"""

    @abstractmethod
    def record_snapshots(self, snapshots: list[Snapshot]):
        """Record a new list of snapshots for every particle. The particle order in the given snapshots list is
        maintained across all calls"""

    @abstractmethod
    def output_data(self):
        """Output current data. This may be called multiple times in the same simulation"""


class TensorboardLogger(Logger):
    def __init__(self, do_log=True, custom_experiment_prefix=""):
        self.do_log = do_log
        self.cur_rollback = 0
        self.num_particles = -1

        self.folder_name = f"runs/{custom_experiment_prefix}_{str(datetime.now()).replace(' ', '_')}"

        self.output_number = 0  # For multiple outputs in the same simulation, differentiates ID

        self.positions: list[list[float]] = []  # vectors of each particle's pos at corresponding self.times index
        self.velocities: list[list[float]] = []  # vectors of each particle's vel at corresponding self.times index
        self.energies: list[list[float]] = []  # vectors of each particle's energy at corresponding self.times index
        self.rollback_numbers: list[int] = []  # stores each datapoints corresponding rollback number
        self.times: list[float] = []  # stores timestamps of data recording

    def rollback(self):
        if not self.do_log: return

        self.cur_rollback += 1

    def record_snapshots(self, snapshots: list[Snapshot]):
        if not self.do_log: return

        # Every snapshot should have the same, so taking it at [0] is arbitrary
        self.times.append(snapshots[0].t)

        self.positions.append([s.x for s in snapshots])
        self.velocities.append([s.v for s in snapshots])
        self.energies.append([s.energy for s in snapshots])

        self.rollback_numbers.append(self.cur_rollback)

        if self.num_particles == -1: self.num_particles = len(snapshots)

    def output_data(self):
        if not self.do_log: return

        writer = SummaryWriter(self.folder_name + f"-{self.output_number}")

        # All the lists should be the same length so choosing self.positions is arbitrary
        for i in range(len(self.positions)):
            for p in range(len(self.positions[i])):
                writer.add_scalar(f"PosP{p}/{self.rollback_numbers[i]}", self.positions[i][p], int(self.times[i] * 100000))
                writer.add_scalar(f"VelP{p}/{self.rollback_numbers[i]}", self.velocities[i][p], int(self.times[i] * 100000))
                writer.add_scalar(f"EnergyP{p}/{self.rollback_numbers[i]}", self.energies[i][p], int(self.times[i] * 100000))

        layout = {
            f"SimulationP{p}": {
                f"PosP{p}": ["Multiline", [f"PosP{p}/{i}" for i in range(self.cur_rollback + 1)]],
                f"VelP{p}": ["Multiline", [f"VelP{p}/{i}" for i in range(self.cur_rollback + 1)]],
                f"EnergyP{p}": ["Multiline", [f"EnergyP{p}/{i}" for i in range(self.cur_rollback + 1)]]
            } for p in range(self.num_particles)
        }

        writer.add_custom_scalars(layout)
        writer.close()

        self.output_number += 1


class SPACM1DSim:
    def __init__(self, rollback_window_size: float, accel_gravity: float, timestep_gravity: float,
                 particle_positions: list[float], particle_velocities: list[float],
                 particle_masses: list[float], collision_stiffness: float, penalty_layer_thickness: float,
                 penalty_timestep: float, logger: Logger, compute_energy: bool = True,
                 gravity_energy_base: float = -10):
        assert len(particle_positions) == len(particle_velocities)

        self.logger = logger

        self.past_events: list[tuple[float, Event]] = []  # History for logging

        self.R = rollback_window_size
        self.penalty_timestep = penalty_timestep

        self.eventQ: list[tuple[float, Event]] = []

        self.accel_gravity = accel_gravity

        def stiffness_lth(l: int):
            return collision_stiffness * (l ** 3)  # ACM paper section 4

        def timestep_lth(l: int, fudge=1e-4):
            return self.penalty_timestep / l / math.sqrt(l + fudge)

        def thickness_lth(l: int):
            return 10 * (l ** (-0.25))

        self.inf_penalty_events = [
            InfinitePenaltyEvent(0, penalty_timestep, stiffness_lth(1)),
            InfinitePenaltyEvent(0 - (thickness_lth(1) - thickness_lth(2)), timestep_lth(2), stiffness_lth(2)),
            InfinitePenaltyEvent(0 - (thickness_lth(1) - thickness_lth(3)), timestep_lth(3), stiffness_lth(3))
        ]

        # Schedule first gravity event
        heapq.heappush(self.eventQ, (timestep_gravity, GravityEvent(timestep_gravity, accel_gravity)))
        # Schedule infinite penalty
        for pen in self.inf_penalty_events:
            heapq.heappush(self.eventQ, (penalty_timestep, pen))

        self.start_of_window = 0
        self.end_of_window = self.R
        self.eventQ_at_start = [(t, e) for t, e in self.eventQ]  # Copies event queue

        self.particles: list[Particle] = [Particle(collision_stiffness, penalty_layer_thickness,
                                                   particle_positions[i], particle_velocities[i],
                                                   0, particle_masses[i]) for i in range(len(particle_positions))]

        self.do_compute_energy = compute_energy  # Should energy be tracked at each snapshot
        self.gravity_base = gravity_energy_base  # Vertical position of "ground" relative to which to compute gravitational energy

    def run_sim(self):
        while True:
            te, e = heapq.heappop(self.eventQ)  # time, event

            for p in self.particles:
                p0 = p.current_snapshots[-1]

                x1 = p0.x + ((te - p0.t) * p0.v)  # integrate position up to this event
                t1 = te  # update to new time

                p.current_snapshots.append(
                    Snapshot(x1, -1, t1, None, False))  # Add new snapshot with placeholder v and has_velocity_changed

            for p in self.particles:
                p0 = p.current_snapshots[-2]
                pc = p.current_snapshots[-1]

                v1_5 = p0.v + (e.h / 2) * (e.get_force(p) / p.mass)  # record energy at halfway vel update
                v1 = p0.v + e.h * (e.get_force(p) / p.mass)  # integrate force with respect to force's timestep

                # Update placeholder v and has_velocity_changed values
                pc.has_velocity_changed = p0.v != v1
                pc.v = v1

                # Update energy
                if self.do_compute_energy:
                    Eg = p.mass * abs(self.accel_gravity) * (pc.x - self.gravity_base)  # E_gravity = mgh
                    Ek = 0.5 * p.mass * (v1_5 ** 2)  # E_kinetic = 1/2 mv^2
                    Ep = 0
                    for pen in self.inf_penalty_events:
                        gap = pen.get_gap(p.get_latest_position())
                        Ep += 0 if gap > 0 else 0.5 * pen.get_stiffness() * (gap ** 2)
                    Etotal = Eg + Ek + Ep
                    pc.energy = Etotal

            # Event-granular log: latest snapshot of all particles
            self.logger.record_snapshots([p.current_snapshots[-1] for p in self.particles])

            heapq.heappush(self.eventQ, (te + e.h, e))  # Schedule next force event

            self.past_events.append((te, e))  # Record processed event

            yield YieldType.EVENT_FINISHED, self

    def output_log_data(self):
        self.logger.output_data()


@dataclass
class Camera:
    """
    world_length_capture: the screen top to bottom captures this length in world space
    world_center: The center of the screen maps to this world position
    """
    world_length_capture: float
    world_center: float

    def world_to_screen_space(self, screen_dim: float, world_pos: float):
        """
        Invert position to match pygame's screen space axes. Scales and positions based on camera

        @param screen_dim: pygame screen height or width depending on which dimension you are plotting along
        @param camera: camera object containing capture details
        @param world_pos: world position of object
        """
        return (screen_dim / 2) - (((world_pos - self.world_center) / self.world_length_capture) * screen_dim)

    def screen_to_world_space(self, screen_dim: float, screen_pos: float):
        return self.world_center + ((((screen_dim / 2) - screen_pos) / screen_dim) * self.world_length_capture)


def get_evenly_spaced_colors(num_colors: int):
    return [tuple([int(val * 255) for val in colorsys.hsv_to_rgb(i * (359 / num_colors), 1, 1)] + [120])
                  for i in range(num_colors)]  # Gets evenly spaced rainbow of colors for each penalty


class PygameVisualizer:
    def __init__(self, screen_size: tuple[int, int], sim: SPACM1DSim, zoom_multiplier: float = 1.1, movement_speed: float = 0.1):
        self.screen = pygame.display.set_mode(screen_size, pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.sim = sim
        self.camera = Camera(10, 0)
        self.screen_size = screen_size
        self.move_speed = movement_speed
        self.zoom_mult = zoom_multiplier
        self.floor_width = 20
        self.penalty_opacity = 120
        self.floor_line_thickness = 3
        self.penalty_line_thickness = 1
        self.font = pygame.font.SysFont("Arial", 11)
        self.settings_font = pygame.font.SysFont("Arial", 13)
        self.timeline_renderer = TimelineVisualizer((self.screen_size[0] // 3, self.screen_size[1]), self.sim)

    def draw_wall(self, col: Collidable):
        h = self.camera.world_to_screen_space(self.screen_size[1], col.get_latest_position())
        center_x = self.screen_size[0] // 2
        pygame.draw.line(self.screen, (0, 0, 0), (center_x - self.floor_width, h), (center_x + self.floor_width, h), self.floor_line_thickness)
        text_name = self.font.render(f"id={col.obj_id}", True, (0, 0, 0))
        self.screen.blit(text_name, (center_x + self.floor_width * 2, h - (text_name.get_height() / 2)))

    def draw_particle(self, snap: Snapshot, particle_id: int):
        circle_pos = (self.screen_size[0] // 2,
                            round(self.camera.world_to_screen_space(self.screen_size[1], snap.x)))
        pygame.draw.circle(self.screen, (255, 0, 0), circle_pos, 5)
        text = self.font.render(f"t={round(snap.t, 3)}", True, (0, 0, 0))
        text_height = self.font.render(f"h={round(snap.x, 3)}", True, (0, 0, 0))
        text_name = self.font.render(f"id={particle_id}", True, (0, 0, 0))
        self.screen.blit(text, (circle_pos[0] + self.floor_width * 2, circle_pos[1] - (text.get_height() / 2)))
        self.screen.blit(text_name, (circle_pos[0] + self.floor_width * 2, circle_pos[1] + (text.get_height() / 2)))
        self.screen.blit(text_height, (circle_pos[0] - self.floor_width * 2 - text.get_width(), circle_pos[1] - (text.get_height() / 2)))

    def draw_penalties(self, col: Collidable):
        num_pen = len(col.active_penalties)
        colors = get_evenly_spaced_colors(num_pen)  # Gets evenly spaced rainbow of colors for each penalty

        for i in range(num_pen):
            thickness = col.get_lth_thickness(col.active_penalties[i].layer)
            top = self.camera.world_to_screen_space(self.screen_size[1], col.get_latest_position() + thickness)
            ctr = self.camera.world_to_screen_space(self.screen_size[1], col.get_latest_position())
            bot = self.camera.world_to_screen_space(self.screen_size[1], col.get_latest_position() - thickness)

            screen_ctr_x = self.screen_size[1] // 2

            surf = pygame.Surface((self.floor_width, abs(top - bot)), pygame.SRCALPHA, 32)
            surf.fill(colors[i])
            self.screen.blit(surf, surf.get_rect(center=(screen_ctr_x, ctr)))
            pygame.draw.line(self.screen, colors[i][:-1], (screen_ctr_x - self.floor_width * 0.7, top),
                             (screen_ctr_x + self.floor_width * 0.7, top), self.penalty_line_thickness)
            pygame.draw.line(self.screen, colors[i][:-1], (screen_ctr_x - self.floor_width * 0.7, bot),
                             (screen_ctr_x + self.floor_width * 0.7, bot), self.penalty_line_thickness)

    def run(self):
        running = True
        s = self.sim.run_sim()

        dragging = None
        auto = False
        show_penalties = True
        granularity = 0
        is_dragging_timeline = False
        do_adjust_timeline = True
        show_timeline = True

        def step():
            typ, _ = next(s)
            # Auto adjust timeline to latest event
            if do_adjust_timeline:
                self.timeline_renderer.set_cam_world_center(
                    self.sim.eventQ[0][0] - (self.timeline_renderer.cam.world_length_capture // 4))

        while running:
            self.screen.fill((255, 255, 255))

            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        running = False
                    elif e.key == pygame.K_RIGHT and not auto:
                        step()
                    elif e.key == pygame.K_a:
                        auto = not auto
                    elif e.key == pygame.K_p:
                        show_penalties = not show_penalties
                    elif e.key == pygame.K_r:
                        granularity = (granularity + 1) % 3
                    elif e.key == pygame.K_o:
                        for p in self.sim.particles:
                            plt.plot([s.t for s in p.previous_snapshots + p.current_snapshots],
                                     [s.x for s in p.previous_snapshots + p.current_snapshots])
                        plt.xlabel("Time")
                        plt.ylabel("Particle Height")
                        plt.show()
                        for p in self.sim.particles:
                            plt.plot([s.t for s in p.previous_snapshots + p.current_snapshots],
                                     [s.energy for s in p.previous_snapshots + p.current_snapshots])
                        plt.xlabel("Time")
                        plt.ylabel("Particle Energy")
                        plt.show()
                        self.sim.output_log_data()
                    elif e.key == pygame.K_t:
                        do_adjust_timeline = not do_adjust_timeline
                    elif e.key == pygame.K_h:
                        show_timeline = not show_timeline

                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == pygame.BUTTON_WHEELDOWN:
                        if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                            self.timeline_renderer.zoom(self.zoom_mult)
                        else:
                            self.camera.world_length_capture *= self.zoom_mult
                    elif e.button == pygame.BUTTON_WHEELUP:
                        if pygame.key.get_pressed()[pygame.K_LSHIFT]:
                            self.timeline_renderer.zoom(1 / self.zoom_mult)
                        else:
                            self.camera.world_length_capture /= self.zoom_mult

                elif e.type == pygame.QUIT:
                    running = False

            if pygame.mouse.get_pressed(3)[0]:
                if dragging is None:
                    is_dragging_timeline = pygame.key.get_pressed()[pygame.K_LSHIFT]
                if dragging is not None:
                    drag_dist = pygame.mouse.get_pos()[1] - dragging
                    if is_dragging_timeline:
                        self.timeline_renderer.drag(drag_dist)

                    else:
                        self.camera.world_center += (drag_dist / self.screen_size[1]) * self.camera.world_length_capture
                dragging = pygame.mouse.get_pos()[1]

            else:
                if dragging is not None:
                    dragging = None

            self.camera.world_center += self.move_speed * self.camera.world_length_capture * (int(pygame.key.get_pressed()[pygame.K_w]) - int(pygame.key.get_pressed()[pygame.K_s]))

            if auto:
                step()

            if show_penalties:
                screen_bottom_pos = self.camera.screen_to_world_space(self.screen_size[1], self.screen_size[1])
                for pen in self.sim.inf_penalty_events:
                    thickness = pen.thresh - screen_bottom_pos
                    if thickness > 0:
                        dummy = Collidable(0, thickness, screen_bottom_pos)
                        dummy.active_penalties.append(PenaltyEvent(0, dummy, 1))
                        self.draw_penalties(dummy)
            for p in self.sim.particles:
                self.draw_particle(p.current_snapshots[-1], p.obj_id)

            if show_timeline:
                self.screen.blit(self.timeline_renderer.render(self.sim), (0, 0))

            padding = 3
            y = padding
            event_granularities = {0: "WINDOW", 1: "ROLLBACK", 2: "EVENT"}
            for text in (
                self.settings_font.render(f"(R) Granularity: {event_granularities[granularity]}", True, (0, 0, 0)),
                self.settings_font.render(f"(P) Show Penalty Layers: {show_penalties}", True, (0, 0, 0)),
                self.settings_font.render(f"(A) Auto Run: {auto}", True, (0, 0, 0)),
                self.settings_font.render(f"(T) Adjust Timeline: {do_adjust_timeline}", True, (0, 0, 0)),
                self.settings_font.render(f"(H) Show Timeline: {show_timeline}", True, (0, 0, 0))
            ):
                self.screen.blit(text, (self.screen_size[0] - text.get_width() - padding, y))
                y += text.get_height() + padding

            pygame.display.update()

            self.clock.tick(120)


class TimelineVisualizer:
    def __init__(self, timeline_surface_dims: tuple[int, int], sim: SPACM1DSim):
        self.cam = Camera(sim.R * 10, 0)
        self.dims = timeline_surface_dims
        self.font = pygame.font.SysFont("Arial", 11)
        self.tick_max_width = 10

    def draw_tick(self, surf: pygame.Surface, screen_pos: tuple[int, int], width: int, thickness: int,
                  color: tuple[int, int, int], label: str, t: float):
        pygame.draw.line(surf, color, (screen_pos[0] - width / 2, screen_pos[1]),
                         (screen_pos[0] + width / 2, screen_pos[1]), thickness)
        text_label = self.font.render(label, True, color)
        text_time = self.font.render(str(round(t, 3)), True, (0, 0, 0))
        surf.blit(text_label, (screen_pos[0] + (width / 2) + 2 + self.tick_max_width,
                               screen_pos[1] - (text_label.get_height() / 2)))
        surf.blit(text_time, (screen_pos[0] - (width / 2) - 2 - text_time.get_width() - self.tick_max_width,
                              screen_pos[1] - (text_label.get_height() / 2)))

    def zoom(self, multiplier: float):
        self.cam.world_length_capture *= multiplier

    def drag(self, screen_space_drag_dist: float):
        self.cam.world_center += (screen_space_drag_dist / self.dims[1]) * self.cam.world_length_capture

    def set_cam_world_center(self, world_center: float):
        self.cam.world_center = world_center

    def render(self, sim: SPACM1DSim) -> pygame.Surface:
        window_size = sim.R
        left_padding = 0

        tick_pos = int(left_padding + (self.font.get_height() * 5) + (self.tick_max_width / 2))

        surf = pygame.Surface(self.dims, pygame.SRCALPHA, 32)
        pygame.draw.line(surf, (0, 0, 0), (tick_pos, 0), (tick_pos, self.dims[1]), 2)

        for t, e in sim.past_events + sim.eventQ:
            pos = self.cam.world_to_screen_space(self.dims[1], t)
            if 0 <= pos <= self.dims[1]:
                if isinstance(e, PenaltyEvent):
                    self.draw_tick(surf, (tick_pos, int(self.cam.world_to_screen_space(self.dims[1], t))),
                                   int(self.tick_max_width // 1.5), 3, (190, 0, 0), f"Pen:L{e.layer}:I{e.owner.obj_id}", t)
                else:
                    color = (0, 0, 190) if isinstance(e, GravityEvent) else (0, 100, 0)
                    self.draw_tick(surf, (tick_pos, int(self.cam.world_to_screen_space(self.dims[1], t))),
                                   int(self.tick_max_width // 1.5), 3, color, f"{type(e).__name__}", t)

        for i in range(int(self.cam.screen_to_world_space(self.dims[1], self.dims[1]) // window_size),
                       int(self.cam.screen_to_world_space(self.dims[1], 0) // window_size) + 1):
            self.draw_tick(surf, (tick_pos, int(self.cam.world_to_screen_space(self.dims[1], i * window_size))),
                           self.tick_max_width, 5, (0, 0, 0), "R", i * window_size)

        if len(sim.past_events):
            pygame.draw.circle(surf, (255, 0, 0), (tick_pos, self.cam.world_to_screen_space(self.dims[1], sim.past_events[-1][0])), 5)

        return surf


if __name__ == "__main__":
    logger = TensorboardLogger(True, "Infinite2LayerDissipationDebug")

    sim = SPACM1DSim(0.03, -2, 0.01, [10], [0], [1], 3, 0, 0.017, logger)

    visualizer = PygameVisualizer((800, 800), sim, 1.1, 0.005)
    visualizer.run()

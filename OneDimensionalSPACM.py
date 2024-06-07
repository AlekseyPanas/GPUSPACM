from __future__ import  annotations
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
pygame.init()


"""
Steps:
- Start sim with params
- Schedule first gravity force event
- Record first time window data
- While events in queue within this time interval R:
    - Pop event
    - Integrate position and update vertex ti, xi
    - Integrate force
    - Schedule next force event if applicable
- Advance to end of time window if space left
- Check missed collisions within time window
- If missed
    - Rollback
    - For each missed collision where object A entered B's next inactive penalty layer
        - Enable B's next penalty layer by scheduling the first force event affecting all objects
- Else
    - Remove any penalty layers exerting 0 force at the end of the time window (detect this by 
    plugging in the position values at the end of the time window into the penalty force equation and seeing if its 0)
"""


@dataclass(unsafe_hash=True)
class Snapshot:
    x: float
    v: float
    t: float
    energy: Optional[float]
    has_velocity_changed: bool


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


class PenaltyEvent(Event):
    """ Single collision penalty layer for a collideable object """
    def __init__(self, timestep: float, owner: Collidable, layer: int):
        super().__init__(timestep)
        self.owner = owner
        self.layer = layer

    def get_force(self, target: Particle) -> float:
        if target is self.owner: return 0

        targ_x = target.get_latest_position()
        own_x = self.owner.get_latest_position()

        gap = abs(own_x - targ_x) - self.owner.get_lth_thickness(self.layer)
        if gap > 0:
            return 0
        else:
            return self.owner.get_lth_stiffness(self.layer) * gap * (-1 if targ_x > own_x else 1)


class Collidable:
    def __init__(self, collision_stiffness: float, first_layer_thickness: float, x: float):
        self.r1 = collision_stiffness  # first penalty layer's collision stiffness
        self.n1 = first_layer_thickness
        self.init_x = x
        self.active_penalties: list[PenaltyEvent] = []

    def get_lth_thickness(self, l: int) -> float:
        return self.n1 * (l ** (-0.25))  # ACM paper section 4

    def get_lth_stiffness(self, l: int) -> float:
        return self.r1 * (l ** 3)  # ACM paper section 4

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


class SPACM1DSim:
    def __init__(self, rollback_window_size: float, accel_gravity: float, timestep_gravity: float,
                 particle_positions: list[float], particle_velocities: list[float], wall_positions: list[float],
                 particle_masses: list[float], collision_stiffness: float, penalty_layer_thickness: float,
                 penalty_timestep: float):
        assert len(particle_positions) == len(particle_velocities)

        self.R = rollback_window_size
        self.penalty_timestep = penalty_timestep

        self.eventQ: list[tuple[float, Event]] = []

        # Schedule first gravity event
        heapq.heappush(self.eventQ, (timestep_gravity, GravityEvent(timestep_gravity, accel_gravity)))

        self.start_of_window = 0
        self.end_of_window = self.R
        self.eventQ_at_start = [(t, e) for t, e in self.eventQ]  # Copies event queue

        self.particles: list[Particle] = [Particle(collision_stiffness, penalty_layer_thickness,
                                                   particle_positions[i], particle_velocities[i],
                                                   0, particle_masses[i]) for i in range(len(particle_positions))]
        self.walls: list[Collidable] = [Collidable(collision_stiffness,
                                                          penalty_layer_thickness,
                                                          p) for p in wall_positions]
        self.collideables: list[Collidable] = self.walls + self.particles

    def run_sim(self):
        while True:
            while self.eventQ[0][0] <= self.end_of_window:
                te, e = heapq.heappop(self.eventQ)  # time, event

                for p in self.particles:
                    p0 = p.current_snapshots[-1]

                    x1 = p0.x + ((te - p0.t) * p0.v)  # integrate position up to this event
                    t1 = te  # update to new time

                    p.current_snapshots.append(Snapshot(x1, -1, t1, None, False))  # Add new snapshot with placeholder v and has_velocity_changed

                for p in self.particles:
                    p0 = p.current_snapshots[-2]
                    pc = p.current_snapshots[-1]
                    v1 = p0.v + e.h * (e.get_force(p) / p.mass)  # integrate force with respect to force's timestep

                    # Update placeholder v and has_velocity_changed values
                    pc.has_velocity_changed = p0.v != v1
                    pc.v = v1

                heapq.heappush(self.eventQ, (te + e.h, e))  # Schedule next force event

                yield YieldType.EVENT_FINISHED, self

            # Linearly update particle positions and timestamps to the end of the rollback window
            for p in self.particles:
                p0 = p.current_snapshots[-1]
                if p0.t < self.end_of_window:
                    x1 = p0.x + ((self.end_of_window - p0.t) * p0.v)
                    t1 = self.end_of_window
                    p.current_snapshots.append(Snapshot(x1, p0.v, t1, None, False))

            # Checks missed collisions
            next_penalty_candidates: list[Collidable] = []  # Objects whose next penalty layer needs activating
            for c in self.collideables:  # Checking c's penalty layers
                move_on = False  # Used as a "break" flag to exit loops once c has been added to the list
                for p in self.particles:  # Moving particles which may have entered a new penalty layer for c
                    if p is not c:  # Don't detect against self, duh

                        # All snapshots where p or c had a velocity change (guaranteed to also include the first and snapshot
                        # of this rollback window)
                        velocity_changed_timestamps = sorted(
                                c.get_velocity_changed_timestamps(self.start_of_window,
                                                                  self.end_of_window).union(
                                p.get_velocity_changed_timestamps(self.start_of_window, self.end_of_window)))

                        # Loop through intervals between above snapshots
                        for i in range(len(velocity_changed_timestamps) - 1):
                            # Get thickness of c's next inactive penalty layer
                            c_next_thickness = c.get_lth_thickness(len(c.active_penalties) + 1)

                            # interval time bounds
                            t0 = velocity_changed_timestamps[i]
                            t1 = velocity_changed_timestamps[i+1]

                            c_positions = [c.get_pos_at_time(t0), c.get_pos_at_time(t1)]
                            p_positions = [p.get_pos_at_time(t0), p.get_pos_at_time(t1)]

                            # Check if particle is already within the penalty layer at start of this interval
                            if c_positions[0] - c_next_thickness <= p_positions[0] <= c_positions[0] + c_next_thickness:
                                next_penalty_candidates.append(c)

                            # Check if there's a t0 <= t <= t1 for which the particle enters the penalty layer
                            else:
                                # Solving two linear equations
                                denom = (c_positions[1] - c_positions[0]) - (p_positions[1] - p_positions[0])
                                if denom != 0:
                                    collision1_t = (p_positions[0] - c_positions[0] - c_next_thickness) / denom
                                    collision2_t = (p_positions[0] - c_positions[0] + c_next_thickness) / denom

                                    if 0 <= collision1_t <= 1 or 0 <= collision2_t <= 1:
                                        next_penalty_candidates.append(c)
                                        move_on = True
                            if move_on: break
                        if move_on: break

            # Initiate rollback
            if len(next_penalty_candidates) > 0:
                # Rollback
                for p in self.particles:
                    p.current_snapshots = [p.current_snapshots[0]]  # Reset snapshots
                    self.eventQ = [(t, e) for t, e in self.eventQ_at_start]  # Restore initial event queue

                # Engage new penalty layers
                for c in next_penalty_candidates:
                    penalty = PenaltyEvent(self.penalty_timestep, c, len(c.active_penalties) + 1)
                    self.eventQ.append((self.start_of_window + penalty.h, penalty))
                    c.active_penalties.append(penalty)

                # Update saved starting event queue
                self.eventQ_at_start = [(t, e) for t, e in self.eventQ]

                yield YieldType.TIME_WINDOW_FINISHED_ROLLBACK, self

            # Proceed to next rollback window
            else:
                # Move old window's snapshots into history
                for p in self.particles:
                    p.previous_snapshots += p.current_snapshots[:-1]
                    p.current_snapshots = [p.current_snapshots[-1]]

                # Remove penalties
                for c in self.collideables:
                    innermost_colliding_layer = None

                    # Finds index of c's innermost layer which is exerting a force on at least one particle
                    for i in range(len(c.active_penalties) - 1,  -1, -1):  # Loop backwards through active penalties
                        anyone_colliding = False

                        for p in self.particles:
                            if c.active_penalties[i].get_force(p) != 0:
                                anyone_colliding = True
                                break

                        if anyone_colliding:
                            innermost_colliding_layer = i
                            break

                    def remove_penalties_from_eventQ(penalties):
                        tups_to_remove = []
                        for tup in self.eventQ:
                            if tup[1] in penalties:
                                tups_to_remove.append(tup)
                        for tup in tups_to_remove:
                            self.eventQ.remove(tup)
                        heapq.heapify(self.eventQ)

                    # No layer is exerting force, remove all
                    if innermost_colliding_layer is None:
                        remove_penalties_from_eventQ(c.active_penalties)
                        c.active_penalties = []
                    # Innermost layer i is exerting force. Keep i and all layers further out than i. Delete the rest
                    else:
                        remove_penalties_from_eventQ(c.active_penalties[innermost_colliding_layer+1:])
                        c.active_penalties = c.active_penalties[:innermost_colliding_layer+1]

                # Update bounds to next window and save event queue state
                self.start_of_window = self.end_of_window
                self.end_of_window = self.start_of_window + self.R
                self.eventQ_at_start = [(t, e) for t, e in self.eventQ]

                yield YieldType.TIME_WINDOW_FINISHED, self


class PygameVisualizer:
    @dataclass
    class Camera:
        """
        world_length_capture: the screen top to bottom captures this length in world space
        world_center: The center of the screen maps to this world position
        """
        world_length_capture: float
        world_center: float

    def __init__(self, screen_size: tuple[int, int], sim: SPACM1DSim, zoom_multiplier: float = 1.1, movement_speed: float = 0.1):
        self.screen = pygame.display.set_mode(screen_size, pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.sim = sim
        self.camera = PygameVisualizer.Camera(10, 0)
        self.screen_size = screen_size
        self.move_speed = movement_speed
        self.zoom_mult = zoom_multiplier
        self.floor_width = 20
        self.penalty_opacity = 120
        self.floor_line_thickness = 3
        self.penalty_line_thickness = 1
        self.font = pygame.font.SysFont("Arial", 11)

    @staticmethod
    def world_to_screen_space(screen_dim: float, camera: PygameVisualizer.Camera, world_pos: float):
        """
        Invert position to match pygame's screen space axes. Scales and positions based on camera

        @param screen_dim: pygame screen height or width depending on which dimension you are plotting along
        @param camera: camera object containing capture details
        @param world_pos: world position of object
        """
        return (screen_dim / 2) - (((world_pos - camera.world_center) / camera.world_length_capture) * screen_dim)

    def draw_wall(self, col: Collidable):
        h = self.world_to_screen_space(self.screen_size[1], self.camera, col.get_latest_position())
        center_x = self.screen_size[0] // 2
        pygame.draw.line(self.screen, (0, 0, 0), (center_x - self.floor_width, h), (center_x + self.floor_width, h), self.floor_line_thickness)

    def draw_particle(self, snap: Snapshot):
        circle_pos = (self.screen_size[0] // 2,
                            round(self.world_to_screen_space(self.screen_size[1], self.camera, snap.x)))
        pygame.draw.circle(self.screen, (255, 0, 0), circle_pos, 5)
        text = self.font.render(f"t={round(snap.t, 3)}", True, (0, 0, 0))
        text_height = self.font.render(f"h={round(snap.x, 3)}", True, (0, 0, 0))
        self.screen.blit(text, (circle_pos[0] + self.floor_width * 2, circle_pos[1] - (text.get_height() / 2)))
        self.screen.blit(text_height, (circle_pos[0] - self.floor_width * 2 - text.get_width(), circle_pos[1] - (text.get_height() / 2)))

    def draw_penalties(self, col: Collidable):
        num_pen = len(col.active_penalties)
        colors = [tuple([int(val * 255) for val in colorsys.hsv_to_rgb(i * (359 / num_pen), 1, 1)] + [120])
                  for i in range(num_pen)]  # Gets evenly spaced rainbow of colors for each penalty

        for i in range(num_pen):
            thickness = col.get_lth_thickness(col.active_penalties[i].layer)
            top = self.world_to_screen_space(self.screen_size[1], self.camera, col.get_latest_position() + thickness)
            ctr = self.world_to_screen_space(self.screen_size[1], self.camera, col.get_latest_position())
            bot = self.world_to_screen_space(self.screen_size[1], self.camera, col.get_latest_position() - thickness)

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
        show_rollbacks = True

        def step():
            # typ, _ = next(s)
            while True:
                typ, _ = next(s)
                if (typ == YieldType.TIME_WINDOW_FINISHED and not show_rollbacks) or \
                        ((typ == YieldType.TIME_WINDOW_FINISHED or typ == YieldType.TIME_WINDOW_FINISHED_ROLLBACK) and show_rollbacks):
                    break

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
                        show_rollbacks = not show_rollbacks
                    elif e.key == pygame.K_o:
                        for p in self.sim.particles:
                            plt.plot([s.x for s in p.previous_snapshots + p.current_snapshots])
                        plt.show()

                elif e.type == pygame.MOUSEBUTTONDOWN:
                    if e.button == pygame.BUTTON_WHEELDOWN:
                        self.camera.world_length_capture *= self.zoom_mult
                    elif e.button == pygame.BUTTON_WHEELUP:
                        self.camera.world_length_capture /= self.zoom_mult

                elif e.type == pygame.QUIT:
                    running = False

            if pygame.mouse.get_pressed(3)[0]:
                if dragging is not None:
                    self.camera.world_center += ((pygame.mouse.get_pos()[1] - dragging) / self.screen_size[1]) * self.camera.world_length_capture
                dragging = pygame.mouse.get_pos()[1]

            else:
                if dragging is not None:
                    dragging = None

            self.camera.world_center += self.move_speed * self.camera.world_length_capture * (int(pygame.key.get_pressed()[pygame.K_w]) - int(pygame.key.get_pressed()[pygame.K_s]))

            if auto:
                step()

            if show_penalties:
                for c in self.sim.collideables:
                    self.draw_penalties(c)
            for p in self.sim.particles:
                self.draw_particle(p.current_snapshots[-1])
            for w in self.sim.walls:
                self.draw_wall(w)

            pygame.display.update()

            self.clock.tick(120)


if __name__ == "__main__":
    # sim = SPACM1DSim(0.03, -1, 0.005, [3], [0], [0], [1], 1, 0.5, 0.005)
    sim = SPACM1DSim(0.03, -1, 0.01, [3, 5], [0, 1], [0], [1, 1], 1, 1, 0.01)  # Working two-particle sim, but bounces are too far
    # sim = SPACM1DSim(0.03, -1, 0.01, [3, 5, 7], [0, 1, 2], [0, 8], [1, 1, 1], 1, 1, 0.01)  # Working three-particle two-wall sim, but bounces are too far

    # sim = SPACM1DSim(0.03, -1, 0.001, [3, 5, 7], [0, 1, 2], [0], [1, 1, 1], 4, 0.3, 0.001)  # Working two-particle sim, bounces not far, but is slow
    # sim = SPACM1DSim(0.03, -1, 0.001, [3, 5, 7], [0, 1, 2], [0, 8], [1, 1, 1], 10, 0.4, 0.001)  # Working three-particle two-wall sim, accurate but slow

    # sim = SPACM1DSim(0.03, -1, 0.03, [10], [0], [0], [1], 1, 0.05, 0.03)  # Infinite loop

    visualizer = PygameVisualizer((800, 800), sim, 1.1, 0.005)
    visualizer.run()

# Goals:
# - Render a set of collideables and particles based on their latest snapshots
# - Camera zoom and move
# - scale ruler for sim
# - snapshot timestamp next to each object
# - step through sim with keys, or run automatically
# - set granularity to event-granular or time-window granular
# - show active penalty layers
# - show next penalty layers




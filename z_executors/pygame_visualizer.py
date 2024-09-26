from __future__ import annotations
from z_sims.core.data import YieldType, Snapshot
from z_sims.core.events import PenaltyEvent
from z_sims.core.sim import Sim
from z_sims.core.objects import Collidable, Particle
import colorsys
from dataclasses import dataclass
import pygame
pygame.init()


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
    def __init__(self, screen_size: tuple[int, int], sim: Sim, zoom_multiplier: float = 1.1, movement_speed: float = 0.1):
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
            while True:
                typ, _ = next(s)
                if (typ == YieldType.TIME_WINDOW_FINISHED and granularity == 0) or \
                        ((typ == YieldType.TIME_WINDOW_FINISHED or typ == YieldType.TIME_WINDOW_FINISHED_ROLLBACK) and granularity == 1) or \
                        (granularity == 2):
                    # Auto adjust timeline to latest event
                    if do_adjust_timeline:
                        self.timeline_renderer.set_cam_world_center(self.sim.get_eventQ()[0][0] - (self.timeline_renderer.cam.world_length_capture // 4))
                    break

        while running:
            self.screen.fill((255, 255, 255))

            for e in pygame.event.get():
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        running = False
                        self.sim.get_logger().quit()
                    elif e.key == pygame.K_RIGHT and not auto:
                        step()
                    elif e.key == pygame.K_a:
                        auto = not auto
                    elif e.key == pygame.K_p:
                        show_penalties = not show_penalties
                    elif e.key == pygame.K_r:
                        granularity = (granularity + 1) % 3
                    elif e.key == pygame.K_o:
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
                for c in self.sim.get_collideables():
                    self.draw_penalties(c)
            for p in self.sim.get_particles():
                self.draw_particle(p.current_snapshots[-1], p.obj_id)
            for w in self.sim.get_walls():
                self.draw_wall(w)

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
                self.settings_font.render(f"(H) Show Timeline: {show_timeline}", True, (0, 0, 0)),
                self.settings_font.render(f"Current Sim Time: {self.sim.get_sim_time()}", True, (0, 0, 0))
            ):
                self.screen.blit(text, (self.screen_size[0] - text.get_width() - padding, y))
                y += text.get_height() + padding

            pygame.display.update()

            self.clock.tick(120)


class TimelineVisualizer:
    def __init__(self, timeline_surface_dims: tuple[int, int], sim: Sim):
        self.cam = Camera(sim.get_window_size() * 10, 0)
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

    def render(self, sim: Sim) -> pygame.Surface:
        window_size = sim.get_window_size()
        left_padding = 0

        tick_pos = int(left_padding + (self.font.get_height() * 5) + (self.tick_max_width / 2))

        surf = pygame.Surface(self.dims, pygame.SRCALPHA, 32)
        pygame.draw.line(surf, (0, 0, 0), (tick_pos, 0), (tick_pos, self.dims[1]), 2)

        for t, tidx, e in sim.get_past_events() + sim.get_eventQ():
            pos = self.cam.world_to_screen_space(self.dims[1], t)
            if 0 <= pos <= self.dims[1]:
                if isinstance(e, PenaltyEvent):
                    self.draw_tick(surf, (tick_pos, int(self.cam.world_to_screen_space(self.dims[1], t))),
                                   int(self.tick_max_width // 1.5), 3, (190, 0, 0), f"Pen:L{e.layer}:I{e.owner.obj_id}", t)
                else:
                    self.draw_tick(surf, (tick_pos, int(self.cam.world_to_screen_space(self.dims[1], t))),
                                   int(self.tick_max_width // 1.5), 3, (0, 0, 190), f"{type(e).__name__}", t)

        for i in range(int(self.cam.screen_to_world_space(self.dims[1], self.dims[1]) // window_size),
                       int(self.cam.screen_to_world_space(self.dims[1], 0) // window_size) + 1):
            self.draw_tick(surf, (tick_pos, int(self.cam.world_to_screen_space(self.dims[1], i * window_size))),
                           self.tick_max_width, 5, (0, 0, 0), "R", i * window_size)

        if len(sim.get_past_events()):
            pygame.draw.circle(surf, (255, 0, 0), (tick_pos, self.cam.world_to_screen_space(self.dims[1], sim.get_past_events()[-1][0])), 5)

        return surf

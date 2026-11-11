#!/usr/bin/env python3
"""
Bouncing Shape & Image Screensaver with Squash, Stretch & Corner Celebration
============================================================================

A minimal, nostalgic bouncing screensaver demonstrating core 2D computer graphics
and real-time animation principles:
- Frame-rate-independent 2D translation using velocity vectors and delta time
- Shape-aware AABB collision detection and vector reflection
- Perfect corner hit detection with numerical tolerance and de-duplication
- Real-time confetti particle system with gravity, drag, rotation, and alpha fade
- Animated congratulatory pop-up banner and non-blocking trumpet fanfare (corner.mp3)
- Non-destructive surface transformations and dimension caching
- Elastic squash-and-stretch procedural animation using damped harmonic easing
- Dynamic viewport resizing, aspect-ratio-preserving scaling, and fullscreen support
- Hidden shape selection menu (S key) with 8 procedural shapes and native image upload

Controls:
- S           : Open hidden shape selection & custom image menu
- SPACE       : Pause / Resume animation
- F11         : Toggle fullscreen mode
- R           : Reset object position, velocity, and deformation state
- C           : Toggle automatic color randomization on wall bounce (for shapes)
- Left Click  : Redirect object velocity toward the clicked cursor position
- Drag & Drop : Drop any PNG/JPG/JPEG/BMP image onto the window to load it
- ESC         : Close shape menu (if open) or exit application
"""

from __future__ import annotations

import argparse
import colorsys
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame

# Optional Pillow import for enhanced image format/alpha normalization if installed
try:
    from PIL import Image as PILImage
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


# =============================================================================
# 1. Configuration & Constants
# =============================================================================

BUILT_IN_SHAPES: Tuple[str, ...] = (
    "Circle",
    "Square",
    "Rectangle",
    "Triangle",
    "Pentagon",
    "Hexagon",
    "Star",
    "Heart",
)

DEFAULT_SHAPE: str = "Circle"
CUSTOM_IMAGE_ID: str = "Custom Image"

# Corner identifiers
CORNER_TOP_LEFT: str = "TOP-LEFT"
CORNER_TOP_RIGHT: str = "TOP-RIGHT"
CORNER_BOTTOM_LEFT: str = "BOTTOM-LEFT"
CORNER_BOTTOM_RIGHT: str = "BOTTOM-RIGHT"

# Celebratory messages displayed on a perfect corner hit
CELEBRATION_MESSAGES: Tuple[str, ...] = (
    "RAAAAAGHHH",
    "YEAAAAAAAAAHHH",
    "POTANGINAAAAA YESSSSSSSSSSS",
    "WABA LUBA DUBDUBBBB",
)

# Vibrant celebratory confetti palette
CONFETTI_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (255, 68, 102),   # Hot Coral Pink
    (255, 214, 10),   # Vivid Gold
    (0, 245, 212),    # Electric Mint Cyan
    (155, 93, 229),   # Vibrant Violet
    (0, 187, 249),    # Sky Neon Blue
    (255, 159, 28),   # Bright Tangerine
    (128, 255, 114),  # Neon Lime
    (241, 91, 181),   # Magenta Rose
)

CORNER_AUDIO_FILENAME: str = "corner.mp3"


@dataclass
class SimulationConfig:
    """Configuration parameters for the Bouncing Screensaver simulation."""
    # Initial window settings
    initial_width: int = 960
    initial_height: int = 600
    min_window_width: int = 320
    min_window_height: int = 240
    target_fps: int = 120
    background_color: Tuple[int, int, int] = (0, 0, 0)  # Pure black

    # Movement & physics settings
    base_speed: float = 280.0          # Speed magnitude in pixels per second
    initial_angle_deg: float = 38.0    # Diagonal launch angle in degrees
    max_delta_time: float = 0.05       # Clamp dt (50ms) to prevent tunneling during window drag

    # Corner collision & celebration settings
    corner_tolerance_px: float = 3.5   # Numerical tolerance (pixels) for simultaneous corner hit
    corner_cooldown_sec: float = 0.15  # Minimum cooldown to prevent duplicate corner triggers
    confetti_min_count: int = 65       # Minimum confetti particles per corner explosion
    confetti_max_count: int = 95       # Maximum confetti particles per corner explosion
    confetti_gravity: float = 420.0    # Downward gravity acceleration (pixels / s^2)
    banner_duration: float = 2.1       # Duration of the congratulatory message (seconds)

    # Squash & Stretch animation parameters
    normal_scale: float = 1.00         # Rest scale factor
    squash_scale: float = 0.75         # Compression along the axis of impact
    stretch_scale: float = 1.20        # Expansion perpendicular to the axis of impact
    recovery_duration: float = 0.26    # Elastic recovery duration in seconds (0.2s - 0.3s)
    elastic_frequency: float = 1.5 * math.pi  # Angular frequency for damped elastic rebound

    # Object sizing & responsive viewport constraints
    nominal_object_size: int = 130     # Nominal bounding-box size for 1:1 shapes
    nominal_image_max_dim: int = 180   # Nominal max dimension for uploaded images / wide shapes
    max_viewport_ratio: float = 0.30   # Max fraction of window width/height object may occupy
    min_object_dimension: int = 24     # Minimum pixel width/height when window is very small

    # Visual behavior
    randomize_color_on_bounce: bool = True
    initial_shape: str = DEFAULT_SHAPE
    custom_image_path: Optional[str] = None


@dataclass(frozen=True)
class CornerHitEvent:
    """Data emitted when the bouncing object achieves a perfect corner hit."""
    corner: str
    position: Tuple[float, float]


# =============================================================================
# 2. Squash and Stretch Procedural Animator
# =============================================================================

class SquashStretchAnimator:
    """
    Manages procedural squash-and-stretch deformation triggered on wall impacts.

    Mathematical Model:
    -------------------
    When the object collides with a boundary at time t = 0:
    1. If hitting a vertical wall (left/right, normal = [+-1, 0]):
       - Horizontal axis compresses: s_x(0) = squash_scale  (0.75)
       - Vertical axis stretches:    s_y(0) = stretch_scale (1.20)
    2. If hitting a horizontal wall (top/bottom, normal = [0, +-1]):
       - Vertical axis compresses:   s_y(0) = squash_scale  (0.75)
       - Horizontal axis stretches:  s_x(0) = stretch_scale (1.20)
    3. If hitting both walls simultaneously (corner hit):
       - Both axes compress symmetrically and rebound.

    Over the recovery duration T (e.g., 0.26 seconds), normalized time is:
        tau = clamp(elapsed_time / T, 0.0, 1.0)

    We interpolate from the initial impact scale `s_0` back to `normal_scale` (1.0)
    using a damped elastic easing curve:
        decay(tau) = (1 - tau)^2 * cos(omega * tau)
        scale(tau) = 1.0 + (s_0 - 1.0) * decay(tau)
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.active: bool = False
        self.elapsed: float = 0.0
        self.duration: float = config.recovery_duration

        # Impact starting scales (s_0 for X and Y axes)
        self.impact_scale_x: float = config.normal_scale
        self.impact_scale_y: float = config.normal_scale

        # Current instantaneous scale factors
        self.scale_x: float = config.normal_scale
        self.scale_y: float = config.normal_scale

    def trigger_impact(self, hit_vertical_wall: bool, hit_horizontal_wall: bool) -> None:
        """
        Trigger squash and stretch along the axis of collision.

        :param hit_vertical_wall: True if collided with Left or Right screen edge.
        :param hit_horizontal_wall: True if collided with Top or Bottom screen edge.
        """
        if not hit_vertical_wall and not hit_horizontal_wall:
            return

        if hit_vertical_wall and hit_horizontal_wall:
            # Corner impact: compress both axes symmetrically
            self.impact_scale_x = self.config.squash_scale
            self.impact_scale_y = self.config.squash_scale
        elif hit_vertical_wall:
            # Left/Right wall: compress horizontally (X), stretch vertically (Y)
            self.impact_scale_x = self.config.squash_scale
            self.impact_scale_y = self.config.stretch_scale
        else:
            # Top/Bottom wall: stretch horizontally (X), compress vertically (Y)
            self.impact_scale_x = self.config.stretch_scale
            self.impact_scale_y = self.config.squash_scale

        self.scale_x = self.impact_scale_x
        self.scale_y = self.impact_scale_y
        self.elapsed = 0.0
        self.active = True

    def reset(self) -> None:
        """Reset deformation immediately to normal 1.0 proportions."""
        self.active = False
        self.elapsed = 0.0
        self.impact_scale_x = self.config.normal_scale
        self.impact_scale_y = self.config.normal_scale
        self.scale_x = self.config.normal_scale
        self.scale_y = self.config.normal_scale

    def update(self, dt: float) -> Tuple[float, float]:
        """
        Advance the squash-and-stretch interpolation by delta time `dt`.

        :param dt: Frame delta time in seconds.
        :return: Tuple of (scale_x, scale_y).
        """
        if not self.active:
            return self.config.normal_scale, self.config.normal_scale

        self.elapsed += dt
        if self.elapsed >= self.duration:
            self.reset()
            return self.scale_x, self.scale_y

        # Normalized progress tau in [0.0, 1.0]
        tau = self.elapsed / self.duration

        # Damped elastic easing factor:
        # Quadratic envelope (1 - tau)^2 guarantees zero value and zero slope at tau = 1,
        # while cos(omega * tau) provides the cartoon-like elastic overshoot & settle.
        envelope = (1.0 - tau) * (1.0 - tau)
        oscillation = math.cos(self.config.elastic_frequency * tau)
        damped_factor = envelope * oscillation

        normal = self.config.normal_scale
        self.scale_x = normal + (self.impact_scale_x - normal) * damped_factor
        self.scale_y = normal + (self.impact_scale_y - normal) * damped_factor

        return self.scale_x, self.scale_y


# =============================================================================
# 3. Procedural Geometric Shape Renderer & Custom Image Loader
# =============================================================================

class AssetManager:
    """
    Generates anti-aliased procedural geometric shapes using Pygame drawing
    functions and loads user images (PNG, JPG, JPEG, BMP) with alpha support.
    """

    @staticmethod
    def create_shape_surface(
        shape_name: str,
        canvas_size: int = 360,
        fill_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    ) -> pygame.Surface:
        """
        Render one of the 8 built-in geometric shapes onto a per-pixel alpha
        surface using 2x supersampling and crop tightly to its visible bounding box.
        White (255, 255, 255) pixels allow clean multiplicative RGB tinting.
        """
        # Supersample at 2x resolution for smooth anti-aliased edges
        ss_size = max(128, canvas_size * 2)
        canvas = pygame.Surface((ss_size, ss_size), pygame.SRCALPHA)
        canvas.fill((0, 0, 0, 0))

        cx = ss_size / 2.0
        cy = ss_size / 2.0
        radius = ss_size * 0.42
        margin = int(ss_size * 0.08)

        shape_key = shape_name.strip().lower()

        if shape_key == "circle":
            pygame.draw.circle(
                canvas,
                fill_color,
                (int(round(cx)), int(round(cy))),
                int(round(radius)),
            )

        elif shape_key == "square":
            side = int(round(radius * 2.0))
            rect = pygame.Rect(0, 0, side, side)
            rect.center = (int(round(cx)), int(round(cy)))
            pygame.draw.rect(canvas, fill_color, rect)

        elif shape_key == "rectangle":
            rect_w = int(round(radius * 2.15))
            rect_h = int(round(radius * 1.30))
            rect = pygame.Rect(0, 0, rect_w, rect_h)
            rect.center = (int(round(cx)), int(round(cy)))
            pygame.draw.rect(canvas, fill_color, rect)

        elif shape_key == "triangle":
            # Upward-pointing equilateral/isosceles triangle
            top_pt = (cx, margin)
            bottom_y = ss_size - margin
            half_base = (bottom_y - margin) / math.sqrt(3.0) * 1.05
            left_pt = (cx - half_base, bottom_y)
            right_pt = (cx + half_base, bottom_y)
            pygame.draw.polygon(canvas, fill_color, [top_pt, right_pt, left_pt])

        elif shape_key == "pentagon":
            points = AssetManager._regular_polygon_points(cx, cy, radius, sides=5, rotation_deg=-90.0)
            pygame.draw.polygon(canvas, fill_color, points)

        elif shape_key == "hexagon":
            points = AssetManager._regular_polygon_points(cx, cy, radius, sides=6, rotation_deg=-90.0)
            pygame.draw.polygon(canvas, fill_color, points)

        elif shape_key == "star":
            points = AssetManager._star_points(
                cx, cy, outer_radius=radius, inner_radius=radius * 0.40, points_count=5
            )
            pygame.draw.polygon(canvas, fill_color, points)

        elif shape_key == "heart":
            points = AssetManager._heart_points(cx, cy, scale=radius / 17.0)
            pygame.draw.polygon(canvas, fill_color, points)

        else:
            # Default fallback: Circle
            pygame.draw.circle(
                canvas,
                fill_color,
                (int(round(cx)), int(round(cy))),
                int(round(radius)),
            )

        # Downsample 2x canvas for anti-aliased edges
        smoothed = pygame.transform.smoothscale(canvas, (canvas_size, canvas_size))

        # Crop tightly to the exact visible bounding box so collision boundaries
        # match the shape's true width and height
        bbox = smoothed.get_bounding_rect()
        if bbox.width > 0 and bbox.height > 0:
            cropped = pygame.Surface(bbox.size, pygame.SRCALPHA)
            cropped.blit(smoothed, (0, 0), bbox)
            return cropped.convert_alpha()

        return smoothed.convert_alpha()

    @staticmethod
    def _regular_polygon_points(
        cx: float, cy: float, radius: float, sides: int, rotation_deg: float = -90.0
    ) -> List[Tuple[float, float]]:
        """Compute vertices for a regular N-sided polygon."""
        pts: List[Tuple[float, float]] = []
        base_angle = math.radians(rotation_deg)
        for i in range(sides):
            theta = base_angle + (2.0 * math.pi * i) / sides
            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)
            pts.append((x, y))
        return pts

    @staticmethod
    def _star_points(
        cx: float,
        cy: float,
        outer_radius: float,
        inner_radius: float,
        points_count: int = 5,
    ) -> List[Tuple[float, float]]:
        """Compute alternating outer and inner vertices for a classic star polygon."""
        pts: List[Tuple[float, float]] = []
        total_vertices = points_count * 2
        base_angle = -math.pi / 2.0  # Point straight up
        for i in range(total_vertices):
            r = outer_radius if (i % 2 == 0) else inner_radius
            theta = base_angle + (math.pi * i) / points_count
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            pts.append((x, y))
        return pts

    @staticmethod
    def _heart_points(
        cx: float, cy: float, scale: float, samples: int = 160
    ) -> List[Tuple[float, float]]:
        """
        Compute vertices for a smooth parametric heart curve:
            x(t) = 16 * sin^3(t)
            y(t) = -(13 * cos(t) - 5 * cos(2t) - 2 * cos(3t) - cos(4t))
        """
        pts: List[Tuple[float, float]] = []
        for i in range(samples):
            t = (2.0 * math.pi * i) / samples
            raw_x = 16.0 * (math.sin(t) ** 3)
            raw_y = -(
                13.0 * math.cos(t)
                - 5.0 * math.cos(2.0 * t)
                - 2.0 * math.cos(3.0 * t)
                - math.cos(4.0 * t)
            )
            pts.append((cx + raw_x * scale, cy + raw_y * scale))
        return pts

    @staticmethod
    def load_image_file(path: Path) -> pygame.Surface:
        """
        Load a user image file (PNG, JPG, JPEG, BMP) into a per-pixel alpha
        Pygame surface, preserving transparency and cropping completely empty
        outer padding so collision bounds match the visible image.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        if HAS_PILLOW:
            with PILImage.open(path) as pil_img:
                rgba_img = pil_img.convert("RGBA")
                bbox = rgba_img.getbbox()
                if bbox:
                    rgba_img = rgba_img.crop(bbox)
                width, height = rgba_img.size
                if width <= 0 or height <= 0:
                    raise ValueError("Image has zero visible dimensions.")

                # Cap overly huge source textures to 1024px max dimension for fast smoothscale
                max_source_dim = 1024
                if width > max_source_dim or height > max_source_dim:
                    scale_factor = max_source_dim / float(max(width, height))
                    new_size = (
                        max(1, int(round(width * scale_factor))),
                        max(1, int(round(height * scale_factor))),
                    )
                    resample_filter = getattr(
                        getattr(PILImage, "Resampling", PILImage),
                        "LANCZOS",
                        PILImage.BICUBIC,
                    )
                    rgba_img = rgba_img.resize(new_size, resample_filter)

                raw_bytes = rgba_img.tobytes()
                surface = pygame.image.fromstring(raw_bytes, rgba_img.size, "RGBA").convert_alpha()
                return surface

        # Standard Pygame image loader fallback
        loaded = pygame.image.load(str(path)).convert_alpha()
        bbox = loaded.get_bounding_rect()
        if bbox.width > 0 and bbox.height > 0:
            cropped = pygame.Surface(bbox.size, pygame.SRCALPHA)
            cropped.blit(loaded, (0, 0), bbox)
            loaded = cropped.convert_alpha()

        w, h = loaded.get_size()
        if w <= 0 or h <= 0:
            raise ValueError("Image has invalid dimensions.")

        max_source_dim = 1024
        if w > max_source_dim or h > max_source_dim:
            scale_factor = max_source_dim / float(max(w, h))
            loaded = pygame.transform.smoothscale(
                loaded,
                (max(1, int(round(w * scale_factor))), max(1, int(round(h * scale_factor)))),
            )
        return loaded

    @staticmethod
    def open_native_file_dialog() -> Optional[Path]:
        """
        Open the OS native file selection dialog for PNG, JPG, JPEG, and BMP files.
        Returns the selected `Path`, or `None` if the user cancels.
        """
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            # Keep the native file dialog above the Pygame window
            try:
                root.attributes("-topmost", True)
            except Exception:
                pass

            selected_file = filedialog.askopenfilename(
                parent=root,
                title="Choose Image from Computer",
                filetypes=[
                    ("Image Files", "*.png *.jpg *.jpeg *.bmp *.PNG *.JPG *.JPEG *.BMP"),
                    ("PNG Images", "*.png *.PNG"),
                    ("JPEG Images", "*.jpg *.jpeg *.JPG *.JPEG"),
                    ("BMP Images", "*.bmp *.BMP"),
                    ("All Files", "*.*"),
                ],
            )
            root.destroy()

            if selected_file:
                return Path(selected_file)
            return None

        except Exception as exc:
            print(f"[WARNING] Native file dialog error: {exc}")
            return None


# =============================================================================
# 4. Universal Bouncing Object (Physics, Corner Detection & Transform Cache)
# =============================================================================

class BouncingObject:
    """
    Universal bouncing entity supporting both procedural geometric shapes and
    custom uploaded images with the same velocity, boundary collision, perfect
    corner detection, and squash-and-stretch engine.

    Computer Graphics Concepts Demonstrated:
    ----------------------------------------
    1. 2D Translation:
       Position vector P = (x, y) is integrated over frame delta time `dt`:
           P(t + dt) = P(t) + V * dt
    2. Shape-Aware Collision Detection, Vector Reflection & Perfect Corner Detection:
       Collision boundaries are calculated from the current object's rendered
       logical dimensions (`base_width`, `base_height`).
       When both horizontal and vertical boundaries are reached simultaneously
       within `corner_tolerance_px`, a `CornerHitEvent` is emitted for the
       specific corner (`TOP-LEFT`, `TOP-RIGHT`, `BOTTOM-LEFT`, `BOTTOM-RIGHT`).
    3. Non-Destructive Image Transformation & Caching:
       `original_surface` is never mutated. Scaling always transforms from
       `active_master_surface` only when target pixel dimensions change.
    """

    def __init__(
        self,
        config: SimulationConfig,
        original_surface: pygame.Surface,
        shape_id: str,
        is_custom_image: bool,
        viewport_size: Tuple[int, int],
    ) -> None:
        self.config = config
        self.animator = SquashStretchAnimator(config)

        # Identity & master surface state
        self.shape_id: str = shape_id
        self.is_custom_image: bool = is_custom_image
        self.original_surface: pygame.Surface = original_surface

        # Aspect ratio of the current shape or uploaded image (width / height)
        orig_w, orig_h = self.original_surface.get_size()
        self.aspect_ratio: float = max(0.05, orig_w / max(1.0, float(orig_h)))

        # Logical base dimensions (before squash/stretch), updated dynamically
        self.base_width: float = float(config.nominal_object_size)
        self.base_height: float = float(config.nominal_object_size)

        # Color state for geometric shapes (uploaded images preserve original appearance)
        self._last_hue: float = random.random()
        self.current_color: Tuple[int, int, int] = self._generate_next_color()
        self.active_master_surface: pygame.Surface = self._build_active_master_surface()

        # Cached scaled surface to avoid redundant smoothscale calls on steady frames
        self._cached_render_size: Tuple[int, int] = (0, 0)
        self._cached_render_surface: Optional[pygame.Surface] = None

        # 2D Position (center of the object) and Velocity vector (pixels / second)
        self.pos = pygame.math.Vector2(viewport_size[0] / 2.0, viewport_size[1] / 2.0)
        self.velocity = pygame.math.Vector2(0.0, 0.0)

        # Cooldown timer preventing duplicate corner celebrations from the same impact
        self._corner_cooldown: float = 0.0

        # Calculate initial responsive size and launch state
        self.update_viewport_constraints(viewport_size)
        self.reset(viewport_size)

    def apply_shape_or_image(
        self,
        new_surface: pygame.Surface,
        shape_id: str,
        is_custom_image: bool,
        viewport_size: Tuple[int, int],
    ) -> None:
        """
        Switch the bouncing object to a new built-in shape or uploaded custom image
        without breaking its position or velocity state, while resetting collision
        deformation and recalculating shape-aware collision bounds.
        """
        self.original_surface = new_surface
        self.shape_id = shape_id
        self.is_custom_image = is_custom_image

        orig_w, orig_h = self.original_surface.get_size()
        self.aspect_ratio = max(0.05, orig_w / max(1.0, float(orig_h)))

        # Reset collision deformation when changing objects as required
        self.animator.reset()
        self._corner_cooldown = 0.0

        # Rebuild master surface and invalidate scale cache
        self._invalidate_surface_cache()

        # Recalculate dimensions and clamp position within current viewport
        self.update_viewport_constraints(viewport_size)

    def _generate_next_color(self) -> Tuple[int, int, int]:
        """
        Pick a new vibrant, high-contrast solid color in HSV space.
        Advances hue by at least 0.18 turns so consecutive colors are always distinct.
        """
        hue_step = random.uniform(0.18, 0.82)
        self._last_hue = (self._last_hue + hue_step) % 1.0
        saturation = random.uniform(0.75, 0.95)
        value = 1.0
        r, g, b = colorsys.hsv_to_rgb(self._last_hue, saturation, value)
        return int(r * 255), int(g * 255), int(b * 255)

    def randomize_color(self) -> None:
        """Select a new random vibrant color for geometric shapes."""
        self.current_color = self._generate_next_color()
        if not self.is_custom_image:
            self._invalidate_surface_cache()

    def _invalidate_surface_cache(self) -> None:
        """Rebuild the active master surface from `original_surface` and clear scale cache."""
        self.active_master_surface = self._build_active_master_surface()
        self._cached_render_size = (0, 0)
        self._cached_render_surface = None

    def _build_active_master_surface(self) -> pygame.Surface:
        """
        For uploaded images, preserve their exact original visual appearance.
        For procedural geometric shapes, apply `self.current_color` non-destructively.
        """
        if self.is_custom_image:
            return self.original_surface.copy()

        tinted = self.original_surface.copy()
        tinted.fill((*self.current_color, 255), special_flags=pygame.BLEND_RGBA_MULT)
        return tinted

    def reset(self, viewport_size: Tuple[int, int]) -> None:
        """Reset object position to the center of the screen and launch diagonally."""
        vw, vh = viewport_size
        self.update_viewport_constraints(viewport_size)
        self.pos.update(vw / 2.0, vh / 2.0)

        angle_rad = math.radians(self.config.initial_angle_deg + random.uniform(-6.0, 6.0))
        dir_x = random.choice([-1.0, 1.0]) * math.cos(angle_rad)
        dir_y = random.choice([-1.0, 1.0]) * math.sin(angle_rad)
        self.velocity.update(dir_x, dir_y)
        self.velocity.scale_to_length(self.config.base_speed)

        self.animator.reset()
        self._corner_cooldown = 0.0
        self._cached_render_size = (0, 0)

    def redirect_toward(self, target_pos: Tuple[int, int]) -> None:
        """
        Point the velocity vector toward `target_pos` (e.g. on mouse click)
        while preserving constant travel speed.
        """
        direction = pygame.math.Vector2(target_pos[0] - self.pos.x, target_pos[1] - self.pos.y)
        if direction.length_squared() < 1e-4:
            self.velocity.rotate_ip(90.0)
        else:
            direction.normalize_ip()
            min_component = 0.25
            if abs(direction.x) < min_component:
                direction.x = math.copysign(min_component, direction.x if direction.x != 0 else 1.0)
            if abs(direction.y) < min_component:
                direction.y = math.copysign(min_component, direction.y if direction.y != 0 else 1.0)
            direction.normalize_ip()
            self.velocity = direction * self.config.base_speed

    def update_viewport_constraints(self, viewport_size: Tuple[int, int]) -> None:
        """
        Calculate the object's logical base dimensions (`base_width`, `base_height`)
        preserving aspect ratio and scaling down automatically if the viewport is small
        or if an uploaded image is oversized.
        """
        vw, vh = max(1, viewport_size[0]), max(1, viewport_size[1])

        # Maximum allowed width and height so the object always has room to bounce
        max_allowed_w = max(self.config.min_object_dimension, vw * self.config.max_viewport_ratio)
        max_allowed_h = max(self.config.min_object_dimension, vh * self.config.max_viewport_ratio)

        # Determine nominal dimensions while preserving exact aspect ratio
        if self.is_custom_image:
            orig_w, orig_h = self.original_surface.get_size()
            scale = min(1.0, self.config.nominal_image_max_dim / float(max(orig_w, orig_h)))
            if max(orig_w, orig_h) < self.config.nominal_object_size:
                scale = self.config.nominal_object_size / float(max(orig_w, orig_h))
            target_w = orig_w * scale
            target_h = orig_h * scale
        else:
            # Built-in geometric shapes: preserve their natural aspect ratio
            if self.aspect_ratio >= 1.0:
                nominal_max = (
                    self.config.nominal_object_size * 1.25
                    if self.aspect_ratio > 1.25
                    else float(self.config.nominal_object_size)
                )
                target_w = nominal_max
                target_h = target_w / self.aspect_ratio
            else:
                target_h = float(self.config.nominal_object_size)
                target_w = target_h * self.aspect_ratio

        # Fit within dynamic viewport ratio constraints
        if target_w > max_allowed_w:
            target_w = max_allowed_w
            target_h = target_w / self.aspect_ratio

        if target_h > max_allowed_h:
            target_h = max_allowed_h
            target_w = target_h * self.aspect_ratio

        # Guarantee minimum pixel dimensions
        min_dim = float(self.config.min_object_dimension)
        if target_w < min_dim and target_h < min_dim:
            if self.aspect_ratio >= 1.0:
                target_w = min_dim
                target_h = max(4.0, target_w / self.aspect_ratio)
            else:
                target_h = min_dim
                target_w = max(4.0, target_h * self.aspect_ratio)

        self.base_width = max(4.0, target_w)
        self.base_height = max(4.0, target_h)

        # Clamp center position so the object stays completely inside the resized window
        self._clamp_position_to_viewport(vw, vh)

    def _clamp_position_to_viewport(self, vw: int, vh: int) -> None:
        """
        Ensure the shape-aware bounding box of the object remains strictly inside
        the visible screen [0, vw] x [0, vh].
        """
        half_w = (self.base_width * max(1.0, self.animator.scale_x)) / 2.0
        half_h = (self.base_height * max(1.0, self.animator.scale_y)) / 2.0

        if vw <= half_w * 2.0:
            self.pos.x = vw / 2.0
        else:
            self.pos.x = max(half_w, min(vw - half_w, self.pos.x))

        if vh <= half_h * 2.0:
            self.pos.y = vh / 2.0
        else:
            self.pos.y = max(half_h, min(vh - half_h, self.pos.y))

    def update(self, dt: float, viewport_size: Tuple[int, int]) -> Optional[CornerHitEvent]:
        """
        Advance physics (2D translation + shape-aware boundary collision reflection)
        and squash-and-stretch animation by frame delta time `dt`.

        :return: `CornerHitEvent` if a perfect corner hit occurred on this update, else `None`.
        """
        vw, vh = viewport_size

        if self._corner_cooldown > 0.0:
            self._corner_cooldown = max(0.0, self._corner_cooldown - dt)

        # 1. 2D Translation: P(t + dt) = P(t) + V * dt
        self.pos.x += self.velocity.x * dt
        self.pos.y += self.velocity.y * dt

        # 2. Shape-Aware Boundary Distances
        half_w = self.base_width / 2.0
        half_h = self.base_height / 2.0
        tol = self.config.corner_tolerance_px

        dist_left = self.pos.x - half_w
        dist_right = vw - (self.pos.x + half_w)
        dist_top = self.pos.y - half_h
        dist_bottom = vh - (self.pos.y + half_h)

        hit_left = (dist_left <= 0.0) and (self.velocity.x < 0.0)
        hit_right = (dist_right <= 0.0) and (self.velocity.x > 0.0)
        hit_top = (dist_top <= 0.0) and (self.velocity.y < 0.0)
        hit_bottom = (dist_bottom <= 0.0) and (self.velocity.y > 0.0)

        # 3. Numerical tolerance check for simultaneous perpendicular boundary contact:
        # If one axis collided on this step and the perpendicular boundary is within
        # `corner_tolerance_px` while moving toward that wall, treat as simultaneous.
        if (hit_left or hit_right) and not (hit_top or hit_bottom):
            if dist_top <= tol and self.velocity.y < 0.0:
                hit_top = True
            elif dist_bottom <= tol and self.velocity.y > 0.0:
                hit_bottom = True
        elif (hit_top or hit_bottom) and not (hit_left or hit_right):
            if dist_left <= tol and self.velocity.x < 0.0:
                hit_left = True
            elif dist_right <= tol and self.velocity.x > 0.0:
                hit_right = True

        # 4. Clamp position and reflect velocity vectors along impacted axes
        if hit_left:
            self.pos.x = half_w
            self.velocity.x = abs(self.velocity.x)
        elif hit_right:
            self.pos.x = max(half_w, vw - half_w)
            self.velocity.x = -abs(self.velocity.x)

        if hit_top:
            self.pos.y = half_h
            self.velocity.y = abs(self.velocity.y)
        elif hit_bottom:
            self.pos.y = max(half_h, vh - half_h)
            self.velocity.y = -abs(self.velocity.y)

        hit_vertical = hit_left or hit_right
        hit_horizontal = hit_top or hit_bottom

        # 5. Trigger Squash-and-Stretch & Color Change on Collision
        corner_event: Optional[CornerHitEvent] = None
        if hit_vertical or hit_horizontal:
            self.animator.trigger_impact(
                hit_vertical_wall=hit_vertical,
                hit_horizontal_wall=hit_horizontal,
            )
            if self.config.randomize_color_on_bounce and not self.is_custom_image:
                self.randomize_color()

            # Check for a legitimate Perfect Corner Hit (both axes hit simultaneously)
            can_move_freely = (vw > self.base_width + tol * 2.0) and (
                vh > self.base_height + tol * 2.0
            )
            if hit_vertical and hit_horizontal and can_move_freely and self._corner_cooldown <= 0.0:
                self._corner_cooldown = self.config.corner_cooldown_sec
                if hit_left and hit_top:
                    corner_event = CornerHitEvent(CORNER_TOP_LEFT, (0.0, 0.0))
                elif hit_right and hit_top:
                    corner_event = CornerHitEvent(CORNER_TOP_RIGHT, (float(vw), 0.0))
                elif hit_left and hit_bottom:
                    corner_event = CornerHitEvent(CORNER_BOTTOM_LEFT, (0.0, float(vh)))
                elif hit_right and hit_bottom:
                    corner_event = CornerHitEvent(CORNER_BOTTOM_RIGHT, (float(vw), float(vh)))

        # 6. Update Squash-and-Stretch Deformation
        self.animator.update(dt)

        return corner_event

    def draw(self, target_surface: pygame.Surface) -> None:
        """
        Transform the current shape or uploaded image from its non-distorted
        master surface according to the current squash-and-stretch scales,
        center it around `self.pos`, and clamp its rendered rect inside the viewport.
        """
        scale_x = self.animator.scale_x
        scale_y = self.animator.scale_y

        # Compute instantaneous pixel dimensions from logical base size and deformation scales
        render_w = max(1, int(round(self.base_width * scale_x)))
        render_h = max(1, int(round(self.base_height * scale_y)))
        target_size = (render_w, render_h)

        # Optimization: Only perform `smoothscale` when pixel dimensions actually change.
        # Always transform from `self.active_master_surface` (never re-scaling a scaled surface).
        if self._cached_render_surface is None or self._cached_render_size != target_size:
            self._cached_render_surface = pygame.transform.smoothscale(
                self.active_master_surface, target_size
            )
            self._cached_render_size = target_size

        # Center the transformed surface around its logical position `self.pos`
        render_rect = self._cached_render_surface.get_rect(
            center=(int(round(self.pos.x)), int(round(self.pos.y)))
        )

        # Guarantee the rendered bounding box never protrudes outside the visible viewport
        vw, vh = target_surface.get_size()
        if render_rect.left < 0:
            render_rect.left = 0
        elif render_rect.right > vw:
            render_rect.right = vw

        if render_rect.top < 0:
            render_rect.top = 0
        elif render_rect.bottom > vh:
            render_rect.bottom = vh

        target_surface.blit(self._cached_render_surface, render_rect)


# =============================================================================
# 5. Perfect Corner Celebration System (Audio, Confetti & Banner)
# =============================================================================

class CornerAudioManager:
    """
    Loads and plays `corner.mp3` using Pygame's mixer when a perfect corner hit
    is detected. Fails gracefully if audio hardware or `corner.mp3` is missing.
    """

    def __init__(self, filename: str = CORNER_AUDIO_FILENAME) -> None:
        self.filename = filename
        self.mixer_ready: bool = False
        self.sound: Optional[pygame.mixer.Sound] = None
        self.use_music_stream: bool = False
        self.audio_path: Optional[Path] = None
        self._warned_missing: bool = False

        self._init_mixer()
        self._try_load_audio()

    def _init_mixer(self) -> None:
        """Initialize pygame.mixer safely without crashing on headless/no-audio systems."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.mixer_ready = True
        except Exception as exc:
            print(f"[WARNING] Audio mixer unavailable: {exc}")
            self.mixer_ready = False

    def _resolve_audio_path(self) -> Optional[Path]:
        """Locate `corner.mp3` next to `main.py` or in the current working directory."""
        candidates = [
            Path(__file__).resolve().parent / self.filename,
            Path(self.filename),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _try_load_audio(self) -> None:
        """Attempt to load `corner.mp3` into Pygame mixer."""
        if not self.mixer_ready:
            return

        path = self._resolve_audio_path()
        if path is None:
            if not self._warned_missing:
                print(
                    f"[INFO] '{self.filename}' not found yet. Celebration visuals will run without sound."
                )
                self._warned_missing = True
            return

        self.audio_path = path
        try:
            self.sound = pygame.mixer.Sound(str(path))
            self.use_music_stream = False
        except Exception:
            # Fallback to pygame.mixer.music stream if Sound() dislikes the MP3 header
            try:
                pygame.mixer.music.load(str(path))
                self.use_music_stream = True
            except Exception as exc:
                print(f"[WARNING] Could not load audio file '{path.name}': {exc}")
                self.sound = None
                self.use_music_stream = False

    def play_celebration_sound(self) -> None:
        """Play `corner.mp3` once without blocking the animation loop."""
        if not self.mixer_ready:
            return

        # If `corner.mp3` was added after launch, attempt lazy load
        if self.sound is None and not self.use_music_stream:
            self._try_load_audio()

        try:
            if self.sound is not None:
                self.sound.stop()
                self.sound.play()
            elif self.use_music_stream and self.audio_path is not None:
                pygame.mixer.music.play()
        except Exception as exc:
            print(f"[WARNING] Audio playback error: {exc}")


@dataclass
class ConfettiParticle:
    """Single celebratory confetti particle (rectangular or circular)."""
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    is_circle: bool
    width: float
    height: float
    angle_deg: float
    angular_vel_deg: float
    age: float
    lifespan: float


class ConfettiSystem:
    """
    Lightweight real-time particle system for corner hit confetti explosions.
    Simulates directional corner bursts, gravity, air resistance, 2D rotation
    for rectangular confetti, and smooth alpha fading.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.particles: List[ConfettiParticle] = []
        # Reusable scratch surface for zero-allocation alpha particle drawing
        self._scratch_size: int = 28
        self._scratch_surf = pygame.Surface(
            (self._scratch_size, self._scratch_size), pygame.SRCALPHA
        )

    def spawn_corner_burst(self, corner: str, position: Tuple[float, float]) -> None:
        """
        Burst 65-95 vibrant confetti particles outward from the impacted corner
        into the interior of the screen.
        """
        origin_x, origin_y = position
        count = random.randint(self.config.confetti_min_count, self.config.confetti_max_count)

        # Determine inward emission angle cone (in degrees, where 0=Right, 90=Down)
        if corner == CORNER_TOP_LEFT:
            min_angle, max_angle = 6.0, 84.0
        elif corner == CORNER_TOP_RIGHT:
            min_angle, max_angle = 96.0, 174.0
        elif corner == CORNER_BOTTOM_LEFT:
            min_angle, max_angle = -84.0, -6.0
        else:  # CORNER_BOTTOM_RIGHT
            min_angle, max_angle = -174.0, -96.0

        for _ in range(count):
            angle_rad = math.radians(random.uniform(min_angle, max_angle))
            speed = random.uniform(180.0, 660.0)
            vx = math.cos(angle_rad) * speed
            vy = math.sin(angle_rad) * speed

            is_circle = random.random() < 0.38
            if is_circle:
                diameter = random.uniform(6.0, 13.0)
                w, h = diameter, diameter
            else:
                w = random.uniform(8.0, 16.0)
                h = random.uniform(4.0, 9.0)

            self.particles.append(
                ConfettiParticle(
                    x=origin_x,
                    y=origin_y,
                    vx=vx,
                    vy=vy,
                    color=random.choice(CONFETTI_PALETTE),
                    is_circle=is_circle,
                    width=w,
                    height=h,
                    angle_deg=random.uniform(0.0, 360.0),
                    angular_vel_deg=random.uniform(-340.0, 340.0),
                    age=0.0,
                    lifespan=random.uniform(2.0, 3.8),
                )
            )

    def update(self, dt: float, viewport_size: Tuple[int, int]) -> None:
        """Advance all active confetti particles using frame delta time `dt`."""
        if not self.particles:
            return

        _, vh = viewport_size
        gravity = self.config.confetti_gravity
        # Exponential air drag so fast initial burst slows into a natural flutter
        drag = math.exp(-1.35 * dt)

        alive: List[ConfettiParticle] = []
        for p in self.particles:
            p.age += dt
            if p.age >= p.lifespan:
                continue

            # Apply air drag and downward gravity
            p.vx *= drag
            p.vy = p.vy * drag + gravity * dt

            p.x += p.vx * dt
            p.y += p.vy * dt
            p.angle_deg = (p.angle_deg + p.angular_vel_deg * dt) % 360.0

            # Keep particle if still above bottom margin of the screen
            if p.y <= vh + 40.0:
                alive.append(p)

        self.particles = alive

    def draw(self, target_surface: pygame.Surface) -> None:
        """Render all active confetti particles with rotation and alpha fading."""
        if not self.particles:
            return

        vw, vh = target_surface.get_size()
        half_s = self._scratch_size / 2.0

        for p in self.particles:
            if p.x < -30.0 or p.x > vw + 30.0 or p.y < -30.0 or p.y > vh + 30.0:
                continue

            # Smooth alpha fade out over the particle's lifespan
            life_ratio = max(0.0, min(1.0, p.age / p.lifespan))
            # Stay bright for the first 35% of life, then fade smoothly to 0
            if life_ratio < 0.35:
                alpha = 255
            else:
                fade_t = (life_ratio - 0.35) / 0.65
                alpha = int(round(255 * (1.0 - fade_t * fade_t)))

            if alpha <= 4:
                continue

            rgba = (*p.color, alpha)
            self._scratch_surf.fill((0, 0, 0, 0))

            if p.is_circle:
                radius = max(2, int(round(p.width * 0.5)))
                pygame.draw.circle(
                    self._scratch_surf,
                    rgba,
                    (int(half_s), int(half_s)),
                    radius,
                )
            else:
                # Compute rotated rectangle vertices around center (half_s, half_s)
                rad = math.radians(p.angle_deg)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                hw = p.width * 0.5
                hh = p.height * 0.5

                dx_w = hw * cos_a
                dy_w = hw * sin_a
                dx_h = -hh * sin_a
                dy_h = hh * cos_a

                pts = [
                    (half_s - dx_w - dx_h, half_s - dy_w - dy_h),
                    (half_s + dx_w - dx_h, half_s + dy_w - dy_h),
                    (half_s + dx_w + dx_h, half_s + dy_w + dy_h),
                    (half_s - dx_w + dx_h, half_s - dy_w + dy_h),
                ]
                pygame.draw.polygon(self._scratch_surf, rgba, pts)

            target_surface.blit(
                self._scratch_surf,
                (int(round(p.x - half_s)), int(round(p.y - half_s))),
            )


class CongratulationsBanner:
    """
    Displays a temporary celebratory message near the center of the screen
    with a pop-in scale animation and smooth alpha fade-out.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.active: bool = False
        self.message: str = ""
        self.elapsed: float = 0.0
        self.duration: float = config.banner_duration
        self._last_message: str = ""

        # Cached rendered text surface for the current active message and viewport width
        self._base_banner_surf: Optional[pygame.Surface] = None
        self._cached_viewport_w: int = 0

    def trigger(self, viewport_size: Tuple[int, int]) -> None:
        """Select a random congratulatory message and start its pop-in animation."""
        choices = [m for m in CELEBRATION_MESSAGES if m != self._last_message] or list(
            CELEBRATION_MESSAGES
        )
        self.message = random.choice(choices)
        self._last_message = self.message
        self.elapsed = 0.0
        self.active = True
        self._rebuild_banner_surface(viewport_size[0])

    def _rebuild_banner_surface(self, viewport_w: int) -> None:
        """Render the bold congratulatory typography with a subtle drop shadow."""
        self._cached_viewport_w = viewport_w
        font_size = max(24, min(68, int(viewport_w * 0.056)))
        font = pygame.font.SysFont(
            ["arialblack", "impact", "segoeui", "arial", "sans"],
            font_size,
            bold=True,
        )

        # Render foreground and dark shadow for high contrast against any confetti/shape
        fg_surf = font.render(self.message, True, (255, 230, 60))
        shadow_surf = font.render(self.message, True, (15, 15, 20))

        pad = 8
        w = fg_surf.get_width() + pad * 2
        h = fg_surf.get_height() + pad * 2
        combined = pygame.Surface((w, h), pygame.SRCALPHA)
        combined.fill((0, 0, 0, 0))

        # Draw multi-directional outline/drop shadow then bright text
        for ox, oy in ((-3, 3), (3, 3), (0, 4), (-2, -2), (2, -2)):
            combined.blit(shadow_surf, (pad + ox, pad + oy))
        combined.blit(fg_surf, (pad, pad))

        # Ensure long celebratory messages always fit comfortably inside the window width
        max_allowed_w = max(120, int(viewport_w * 0.88))
        if w > max_allowed_w:
            fit_scale = max_allowed_w / float(w)
            combined = pygame.transform.smoothscale(
                combined,
                (max(1, int(round(w * fit_scale))), max(1, int(round(h * fit_scale)))),
            )

        self._base_banner_surf = combined.convert_alpha()

    def update(self, dt: float) -> None:
        """Advance the banner timer."""
        if not self.active:
            return

        self.elapsed += dt
        if self.elapsed >= self.duration:
            self.active = False
            self._base_banner_surf = None

    def draw(self, target_surface: pygame.Surface) -> None:
        """Render the pop-in scaled and alpha-faded congratulatory message."""
        if not self.active or not self.message:
            return

        vw, vh = target_surface.get_size()
        if self._base_banner_surf is None or abs(vw - self._cached_viewport_w) > 40:
            self._rebuild_banner_surface(vw)

        assert self._base_banner_surf is not None

        # 1. Pop-in elastic scale during first 0.24 seconds
        pop_duration = 0.24
        if self.elapsed < pop_duration:
            u = self.elapsed / pop_duration
            # Overshoot curve: 0.35 -> 1.14 -> 1.0
            scale = 1.0 - 0.65 * ((1.0 - u) ** 2) * math.cos(1.35 * math.pi * u)
        else:
            scale = 1.0

        # 2. Smooth fade-out during the final 0.55 seconds
        fade_duration = 0.55
        time_left = self.duration - self.elapsed
        if time_left < fade_duration:
            fade_ratio = max(0.0, min(1.0, time_left / fade_duration))
            alpha = int(round(255 * (fade_ratio * fade_ratio)))
        else:
            alpha = 255

        if alpha <= 2:
            return

        base_w, base_h = self._base_banner_surf.get_size()
        if abs(scale - 1.0) > 0.01:
            scaled_w = max(1, int(round(base_w * scale)))
            scaled_h = max(1, int(round(base_h * scale)))
            render_surf = pygame.transform.smoothscale(
                self._base_banner_surf, (scaled_w, scaled_h)
            )
        else:
            render_surf = self._base_banner_surf.copy()

        if alpha < 255:
            render_surf.set_alpha(alpha)

        rect = render_surf.get_rect(center=(vw // 2, int(vh * 0.46)))
        target_surface.blit(render_surf, rect)


class CelebrationManager:
    """
    Coordinates the trumpet audio (`corner.mp3`), confetti particle explosion,
    and congratulatory banner whenever a perfect corner hit occurs.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.audio = CornerAudioManager(CORNER_AUDIO_FILENAME)
        self.confetti = ConfettiSystem(config)
        self.banner = CongratulationsBanner(config)
        self.total_corner_hits: int = 0

    def trigger_celebration(
        self, event: CornerHitEvent, viewport_size: Tuple[int, int]
    ) -> None:
        """
        Activate all celebration effects simultaneously on a perfect corner hit:
        1. Play `corner.mp3` trumpet sound once
        2. Burst confetti particles from the impacted corner
        3. Pop up a random congratulatory message near the screen center
        """
        self.total_corner_hits += 1
        self.audio.play_celebration_sound()
        self.confetti.spawn_corner_burst(event.corner, event.position)
        self.banner.trigger(viewport_size)

    def update(self, dt: float, viewport_size: Tuple[int, int]) -> None:
        """Advance confetti particles and congratulatory banner independently."""
        self.confetti.update(dt, viewport_size)
        self.banner.update(dt)

    def draw(self, target_surface: pygame.Surface) -> None:
        """Render active confetti and the congratulatory message."""
        self.confetti.draw(target_surface)
        self.banner.draw(target_surface)


# =============================================================================
# 6. Hidden Shape Selection & Custom Image Menu (Modular UI Component)
# =============================================================================

@dataclass
class MenuActionResult:
    """Result returned by `ShapeSelectionMenu` after handling an event."""
    close_menu: bool = False
    selected_shape: Optional[str] = None
    request_file_upload: bool = False


class ShapeSelectionMenu:
    """
    Hidden customization overlay activated by pressing the 'S' key.

    Features:
    - Displays a clean 4x2 grid of the 8 built-in shapes with visual previews
    - Highlights the currently active shape (or custom image status)
    - Provides a bottom button labeled "Choose Image from Computer"
    - Dynamically adapts its layout to any window size
    - Closes on ESC without changing the current selection
    """

    UPLOAD_BUTTON_LABEL = "Choose Image from Computer"

    def __init__(self, shape_surfaces: Dict[str, pygame.Surface]) -> None:
        self.is_open: bool = False
        self.error_message: Optional[str] = None

        # Pre-generate crisp preview icons for each built-in shape
        self.preview_cache: Dict[Tuple[str, int], pygame.Surface] = {}
        self.master_shapes: Dict[str, pygame.Surface] = shape_surfaces

        # Interactive hit rects updated on draw/layout
        self.card_rects: Dict[str, pygame.Rect] = {}
        self.upload_button_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)

        # Fonts (initialized lazily after pygame.font.init())
        self._title_font: Optional[pygame.font.Font] = None
        self._label_font: Optional[pygame.font.Font] = None
        self._button_font: Optional[pygame.font.Font] = None
        self._hint_font: Optional[pygame.font.Font] = None
        self._cached_font_scale: int = 0

    def open(self) -> None:
        """Open the shape selection menu."""
        self.is_open = True
        self.error_message = None

    def close(self) -> None:
        """Close the shape selection menu."""
        self.is_open = False
        self.error_message = None

    def set_error(self, message: str) -> None:
        """Display a non-fatal error message (e.g., if image loading fails)."""
        self.error_message = message

    def _ensure_fonts(self, modal_h: int) -> None:
        """Create or resize UI fonts proportionally to the modal height."""
        scale_bucket = max(12, min(32, modal_h // 20))
        if self._title_font is not None and scale_bucket == self._cached_font_scale:
            return

        self._cached_font_scale = scale_bucket
        font_names = ["segoeui", "arial", "calibri", "verdana", "sans"]
        self._title_font = pygame.font.SysFont(font_names, int(scale_bucket * 1.15), bold=True)
        self._label_font = pygame.font.SysFont(font_names, max(11, int(scale_bucket * 0.72)), bold=True)
        self._button_font = pygame.font.SysFont(font_names, max(12, int(scale_bucket * 0.80)), bold=True)
        self._hint_font = pygame.font.SysFont(font_names, max(10, int(scale_bucket * 0.62)))

    def _get_preview_surface(
        self, shape_name: str, max_box_size: int, tint_color: Tuple[int, int, int]
    ) -> pygame.Surface:
        """Return a scaled, aspect-ratio-preserving preview surface for `shape_name`."""
        box_size = max(12, max_box_size)
        key = (shape_name, box_size)
        if key not in self.preview_cache:
            master = self.master_shapes[shape_name]
            mw, mh = master.get_size()
            aspect = mw / max(1.0, float(mh))
            if aspect >= 1.0:
                pw = box_size
                ph = max(1, int(round(box_size / aspect)))
            else:
                ph = box_size
                pw = max(1, int(round(box_size * aspect)))
            self.preview_cache[key] = pygame.transform.smoothscale(master, (pw, ph))

        tinted = self.preview_cache[key].copy()
        tinted.fill((*tint_color, 255), special_flags=pygame.BLEND_RGBA_MULT)
        return tinted

    def _compute_layout(self, viewport_size: Tuple[int, int]) -> Tuple[pygame.Rect, int, int]:
        """Compute responsive modal, grid card, and upload button rectangles."""
        vw, vh = viewport_size

        modal_w = max(280, min(680, int(vw * 0.84)))
        modal_h = max(220, min(500, int(vh * 0.86)))
        modal_rect = pygame.Rect(0, 0, modal_w, modal_h)
        modal_rect.center = (vw // 2, vh // 2)

        pad_x = max(12, int(modal_w * 0.045))
        pad_y = max(10, int(modal_h * 0.045))

        header_h = max(36, int(modal_h * 0.14))
        button_h = max(34, min(52, int(modal_h * 0.115)))
        footer_area_h = button_h + pad_y * 2

        grid_top = modal_rect.top + header_h + pad_y // 2
        grid_bottom = modal_rect.bottom - footer_area_h
        grid_h = max(60, grid_bottom - grid_top)
        grid_w = modal_w - 2 * pad_x

        cols = 4
        rows = 2
        gap_x = max(8, int(grid_w * 0.025))
        gap_y = max(8, int(grid_h * 0.06))

        card_w = max(44, (grid_w - gap_x * (cols - 1)) // cols)
        card_h = max(36, (grid_h - gap_y * (rows - 1)) // rows)

        # Center the grid horizontally inside the modal
        total_grid_w = card_w * cols + gap_x * (cols - 1)
        grid_left = modal_rect.left + (modal_w - total_grid_w) // 2

        self.card_rects.clear()
        for idx, shape_name in enumerate(BUILT_IN_SHAPES):
            r = idx // cols
            c = idx % cols
            x = grid_left + c * (card_w + gap_x)
            y = grid_top + r * (card_h + gap_y)
            self.card_rects[shape_name] = pygame.Rect(x, y, card_w, card_h)

        # Upload button centered at the bottom of the modal
        btn_w = min(total_grid_w, max(220, int(modal_w * 0.72)))
        btn_x = modal_rect.centerx - btn_w // 2
        btn_y = modal_rect.bottom - pad_y - button_h
        self.upload_button_rect = pygame.Rect(btn_x, btn_y, btn_w, button_h)

        return modal_rect, header_h, pad_y

    def handle_event(self, event: pygame.event.Event, viewport_size: Tuple[int, int]) -> MenuActionResult:
        """Process keyboard and mouse input while the shape selection menu is open."""
        self._compute_layout(viewport_size)

        if event.type == pygame.KEYDOWN:
            # Pressing ESC closes the menu without changing the currently selected shape
            if event.key == pygame.K_ESCAPE:
                return MenuActionResult(close_menu=True)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos

            # Check if a built-in shape card was clicked
            for shape_name, rect in self.card_rects.items():
                if rect.collidepoint(mouse_pos):
                    return MenuActionResult(
                        close_menu=True,
                        selected_shape=shape_name,
                    )

            # Check if "Choose Image from Computer" button was clicked
            if self.upload_button_rect.collidepoint(mouse_pos):
                return MenuActionResult(request_file_upload=True)

        return MenuActionResult()

    def draw(
        self,
        target_surface: pygame.Surface,
        active_shape_id: str,
        is_custom_image: bool,
    ) -> None:
        """Render the semi-transparent backdrop, modal dialog, shape grid, and upload button."""
        if not self.is_open:
            return

        vw, vh = target_surface.get_size()
        modal_rect, header_h, pad_y = self._compute_layout((vw, vh))
        self._ensure_fonts(modal_rect.height)

        # 1. Dimmed translucent overlay over the paused animation screen
        overlay = pygame.Surface((vw, vh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 195))
        target_surface.blit(overlay, (0, 0))

        # 2. Modal dialog container
        pygame.draw.rect(target_surface, (22, 24, 30), modal_rect, border_radius=14)
        pygame.draw.rect(target_surface, (72, 80, 100), modal_rect, width=2, border_radius=14)

        # 3. Header title & ESC instructions
        assert self._title_font and self._hint_font and self._label_font and self._button_font
        title_surf = self._title_font.render("Select a Shape", True, (242, 245, 250))
        title_rect = title_surf.get_rect(
            centerx=modal_rect.centerx,
            top=modal_rect.top + max(8, pad_y // 2),
        )
        target_surface.blit(title_surf, title_rect)

        subtitle_text = (
            self.error_message
            if self.error_message
            else "Click a shape to apply  •  Press ESC to cancel"
        )
        subtitle_color = (255, 110, 110) if self.error_message else (150, 160, 180)
        hint_surf = self._hint_font.render(subtitle_text, True, subtitle_color)
        hint_rect = hint_surf.get_rect(
            centerx=modal_rect.centerx,
            top=title_rect.bottom + 2,
        )
        target_surface.blit(hint_surf, hint_rect)

        mouse_pos = pygame.mouse.get_pos()

        # 4. Draw 4x2 grid of shape preview cards
        for shape_name, card_rect in self.card_rects.items():
            is_selected = (not is_custom_image) and (shape_name == active_shape_id)
            is_hovered = card_rect.collidepoint(mouse_pos)

            if is_selected:
                bg_color = (34, 58, 92)
                border_color = (88, 178, 255)
                border_width = 3
                preview_tint = (110, 200, 255)
                text_color = (255, 255, 255)
            elif is_hovered:
                bg_color = (38, 42, 54)
                border_color = (140, 155, 185)
                border_width = 2
                preview_tint = (240, 244, 255)
                text_color = (240, 244, 255)
            else:
                bg_color = (30, 33, 42)
                border_color = (58, 64, 80)
                border_width = 1
                preview_tint = (200, 208, 224)
                text_color = (195, 202, 218)

            pygame.draw.rect(target_surface, bg_color, card_rect, border_radius=10)
            pygame.draw.rect(
                target_surface, border_color, card_rect, width=border_width, border_radius=10
            )

            # Label at bottom of card
            label_surf = self._label_font.render(shape_name, True, text_color)
            label_rect = label_surf.get_rect(
                centerx=card_rect.centerx,
                bottom=card_rect.bottom - max(5, card_rect.height // 12),
            )
            target_surface.blit(label_surf, label_rect)

            # Shape visual preview centered in upper portion of card
            avail_h = max(14, label_rect.top - card_rect.top - 10)
            avail_w = max(14, card_rect.width - 16)
            preview_box = min(avail_w, avail_h)
            preview_surf = self._get_preview_surface(shape_name, preview_box, preview_tint)
            preview_rect = preview_surf.get_rect(
                center=(card_rect.centerx, card_rect.top + (label_rect.top - card_rect.top) // 2)
            )
            target_surface.blit(preview_surf, preview_rect)

        # 5. Draw "Choose Image from Computer" button at the bottom
        btn_hovered = self.upload_button_rect.collidepoint(mouse_pos)
        if is_custom_image:
            btn_bg = (42, 78, 120) if not btn_hovered else (54, 96, 146)
            btn_border = (95, 190, 255)
            btn_border_w = 2
        elif btn_hovered:
            btn_bg = (52, 88, 138)
            btn_border = (130, 195, 255)
            btn_border_w = 2
        else:
            btn_bg = (38, 64, 104)
            btn_border = (82, 130, 195)
            btn_border_w = 2

        pygame.draw.rect(target_surface, btn_bg, self.upload_button_rect, border_radius=9)
        pygame.draw.rect(
            target_surface,
            btn_border,
            self.upload_button_rect,
            width=btn_border_w,
            border_radius=9,
        )

        btn_text_surf = self._button_font.render(
            self.UPLOAD_BUTTON_LABEL, True, (245, 248, 255)
        )
        btn_text_rect = btn_text_surf.get_rect(center=self.upload_button_rect.center)
        target_surface.blit(btn_text_surf, btn_text_rect)


# =============================================================================
# 7. Application Controller (Windowing, Event Loop & Menu/Celebration Integration)
# =============================================================================

class BouncingScreensaverApp:
    """
    Main Pygame application managing window lifecycle, dynamic resizing,
    fullscreen toggling, hidden shape selection menu, corner celebration
    effects, and real-time animation.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        pygame.init()
        pygame.font.init()

        pygame.display.set_caption("Bouncing Screensaver")

        # Window state tracking for seamless fullscreen toggling
        self.is_fullscreen: bool = False
        self.windowed_size: Tuple[int, int] = (
            config.initial_width,
            config.initial_height,
        )

        self.screen: pygame.Surface = pygame.display.set_mode(
            self.windowed_size,
            pygame.RESIZABLE | pygame.DOUBLEBUF,
        )

        # Pre-generate high-resolution master surfaces for all 8 built-in shapes
        self.shape_surfaces: Dict[str, pygame.Surface] = {
            name: AssetManager.create_shape_surface(name) for name in BUILT_IN_SHAPES
        }

        # Determine initial object (Default: solid-colored Circle, or CLI custom image if given)
        initial_surface = self.shape_surfaces[DEFAULT_SHAPE]
        initial_id = DEFAULT_SHAPE
        is_custom = False

        if config.custom_image_path:
            try:
                initial_surface = AssetManager.load_image_file(Path(config.custom_image_path))
                initial_id = CUSTOM_IMAGE_ID
                is_custom = True
            except Exception as exc:
                print(f"[WARNING] Failed to load '{config.custom_image_path}': {exc}. Using default Circle.")

        # Set window icon from the initial surface
        icon_surface = pygame.transform.smoothscale(initial_surface, (32, 32))
        pygame.display.set_icon(icon_surface)

        self.bouncer = BouncingObject(
            config=config,
            original_surface=initial_surface,
            shape_id=initial_id,
            is_custom_image=is_custom,
            viewport_size=self.screen.get_size(),
        )

        # Modular hidden shape selection menu
        self.menu = ShapeSelectionMenu(self.shape_surfaces)

        # Modular perfect corner hit celebration system (audio + confetti + banner)
        self.celebration = CelebrationManager(config)

        self.clock = pygame.time.Clock()
        self.paused: bool = False
        self._paused_before_menu: bool = False
        self.running: bool = True

    def open_shape_menu(self) -> None:
        """Open the hidden shape menu and pause the animation."""
        if not self.menu.is_open:
            self._paused_before_menu = self.paused
            self.paused = True
            self.menu.open()

    def close_shape_menu(self, resume_animation: bool = False) -> None:
        """Close the shape selection menu and restore or resume animation."""
        if self.menu.is_open:
            self.menu.close()
            if resume_animation:
                self.paused = False
            else:
                self.paused = self._paused_before_menu
            # Flush accumulated frame delta time while in menu/dialog
            self.clock.tick()

    def select_builtin_shape(self, shape_name: str) -> None:
        """Apply one of the 8 built-in geometric shapes to the bouncing object."""
        if shape_name not in self.shape_surfaces:
            return

        self.bouncer.apply_shape_or_image(
            new_surface=self.shape_surfaces[shape_name],
            shape_id=shape_name,
            is_custom_image=False,
            viewport_size=self.screen.get_size(),
        )
        self.close_shape_menu(resume_animation=True)

    def prompt_and_load_custom_image(self) -> None:
        """
        Open the native computer file dialog to select a PNG, JPG, JPEG, or BMP
        image. If canceled, return to the shape selection menu without changes.
        """
        selected_path = AssetManager.open_native_file_dialog()
        # Reset clock tick so time spent inside OS file dialog does not jump the simulation
        self.clock.tick()

        if selected_path is None:
            # User canceled file selection: stay in the shape selection menu without changing anything
            return

        self._try_apply_custom_image(selected_path, close_menu_on_success=True)

    def _try_apply_custom_image(
        self, image_path: Path, close_menu_on_success: bool = True
    ) -> bool:
        """Load an image from `image_path` and set it as the active bouncing object."""
        try:
            loaded_surface = AssetManager.load_image_file(image_path)
            self.bouncer.apply_shape_or_image(
                new_surface=loaded_surface,
                shape_id=CUSTOM_IMAGE_ID,
                is_custom_image=True,
                viewport_size=self.screen.get_size(),
            )
            if close_menu_on_success and self.menu.is_open:
                self.close_shape_menu(resume_animation=True)
            return True
        except Exception as exc:
            msg = f"Could not load image '{image_path.name}'"
            print(f"[WARNING] {msg}: {exc}")
            if self.menu.is_open:
                self.menu.set_error(msg)
            return False

    def toggle_fullscreen(self) -> None:
        """Toggle between windowed resizable mode and fullscreen mode (F11)."""
        self.is_fullscreen = not self.is_fullscreen

        if self.is_fullscreen:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode(
                (0, 0),
                pygame.FULLSCREEN | pygame.DOUBLEBUF,
            )
        else:
            self.screen = pygame.display.set_mode(
                self.windowed_size,
                pygame.RESIZABLE | pygame.DOUBLEBUF,
            )

        self.bouncer.update_viewport_constraints(self.screen.get_size())

    def handle_resize(self, width: int, height: int) -> None:
        """Handle dynamic window resize events and recalculate collision boundaries."""
        if self.is_fullscreen:
            return

        clamped_w = max(self.config.min_window_width, width)
        clamped_h = max(self.config.min_window_height, height)
        self.windowed_size = (clamped_w, clamped_h)

        current_w, current_h = self.screen.get_size()
        if (current_w, current_h) != (clamped_w, clamped_h):
            self.screen = pygame.display.set_mode(
                (clamped_w, clamped_h),
                pygame.RESIZABLE | pygame.DOUBLEBUF,
            )

        self.bouncer.update_viewport_constraints(self.screen.get_size())

    def handle_events(self) -> None:
        """Process keyboard, mouse, window resize, menu interaction, and drag-and-drop events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            # Window resizing and fullscreen are always handled
            if event.type == pygame.VIDEORESIZE:
                self.handle_resize(event.w, event.h)
                continue

            if hasattr(pygame, "WINDOWSIZECHANGED") and event.type == pygame.WINDOWSIZECHANGED:
                self.handle_resize(event.x, event.y)
                continue

            if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                self.toggle_fullscreen()
                continue

            # Support drag-and-drop image replacement at any time
            if event.type == pygame.DROPFILE:
                self._try_apply_custom_image(Path(event.file), close_menu_on_success=True)
                continue

            # Route input to the Shape Selection Menu when it is open
            if self.menu.is_open:
                result = self.menu.handle_event(event, self.screen.get_size())
                if result.selected_shape:
                    self.select_builtin_shape(result.selected_shape)
                elif result.request_file_upload:
                    self.prompt_and_load_custom_image()
                elif result.close_menu:
                    self.close_shape_menu(resume_animation=False)
                continue

            # Standard simulation controls when menu is closed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_s:
                    self.open_shape_menu()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.bouncer.reset(self.screen.get_size())
                elif event.key == pygame.K_c:
                    self.config.randomize_color_on_bounce = (
                        not self.config.randomize_color_on_bounce
                    )

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.bouncer.redirect_toward(event.pos)

    def run(self) -> None:
        """Main real-time simulation loop."""
        while self.running:
            raw_dt = self.clock.tick(self.config.target_fps) / 1000.0
            dt = min(raw_dt, self.config.max_delta_time)

            self.handle_events()

            viewport_size = self.screen.get_size()
            self.bouncer.update_viewport_constraints(viewport_size)

            if not self.paused and not self.menu.is_open:
                corner_hit = self.bouncer.update(dt, viewport_size)
                if corner_hit is not None:
                    self.celebration.trigger_celebration(corner_hit, viewport_size)
                self.celebration.update(dt, viewport_size)

            # Render pure black background, bouncing object, and active celebration effects
            self.screen.fill(self.config.background_color)
            self.bouncer.draw(self.screen)
            self.celebration.draw(self.screen)

            # Render hidden shape selection menu overlay only when active
            if self.menu.is_open:
                self.menu.draw(
                    self.screen,
                    active_shape_id=self.bouncer.shape_id,
                    is_custom_image=self.bouncer.is_custom_image,
                )

            pygame.display.flip()

        pygame.quit()


# =============================================================================
# 8. CLI Argument Parsing & Entry Point
# =============================================================================

def parse_args() -> SimulationConfig:
    """Parse optional command-line arguments into a `SimulationConfig` instance."""
    parser = argparse.ArgumentParser(
        description="Bouncing Shape & Image Screensaver with Squash, Stretch & Corner Celebration"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Optional path to a custom image file (PNG, JPG, JPEG, BMP).",
    )
    parser.add_argument(
        "-i",
        "--image",
        dest="image_flag",
        default=None,
        help="Path to a custom image file (PNG, JPG, JPEG, BMP).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=280.0,
        help="Movement speed of the object in pixels per second (default: 280).",
    )
    parser.add_argument(
        "--no-color-change",
        action="store_true",
        help="Disable randomizing the shape's color when bouncing off walls.",
    )
    parser.add_argument(
        "--squash",
        type=float,
        default=0.75,
        help="Squash scale along impact axis (default: 0.75).",
    )
    parser.add_argument(
        "--stretch",
        type=float,
        default=1.20,
        help="Stretch scale perpendicular to impact axis (default: 0.20 -> 1.20).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.26,
        help="Squash-and-stretch recovery duration in seconds (default: 0.26).",
    )

    args = parser.parse_args()
    custom_image = args.image_flag or args.image

    return SimulationConfig(
        base_speed=max(20.0, args.speed),
        squash_scale=max(0.2, min(1.0, args.squash)),
        stretch_scale=max(1.0, min(2.5, args.stretch)),
        recovery_duration=max(0.05, min(2.0, args.duration)),
        randomize_color_on_bounce=not args.no_color_change,
        custom_image_path=custom_image,
    )


def main() -> None:
    # Enable DPI awareness on Windows for crisp rendering at native resolution
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    config = parse_args()
    app = BouncingScreensaverApp(config)
    app.run()


if __name__ == "__main__":
    main()

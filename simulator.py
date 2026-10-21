"""
Virtual Combat Arena Simulation Engine.
Generates realistic corner camera perspectives with moving robots, ArUco markers,
and arena obstacles to test and demonstrate the full tracking & steering pipeline.
"""

import math
import cv2
import numpy as np

class ArenaSimulator:
    def __init__(self, arena_size: int = 800):
        self.arena_size = arena_size
        
        # Load or generate ArUco marker image
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, 'generateImageMarker'):
            self.tag_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 50)
        else:
            self.tag_img = cv2.aruco.drawMarker(aruco_dict, 0, 50)
        self.tag_img = cv2.cvtColor(self.tag_img, cv2.COLOR_GRAY2BGR)

        # Robot states in arena floor coordinates (0 to arena_size)
        self.my_x = 250.0
        self.my_y = 550.0
        self.my_heading = -45.0  # degrees
        
        self.enemy_x = 550.0
        self.enemy_y = 250.0
        self.enemy_vx = 2.5
        self.enemy_vy = 1.8
        
        # Occlusion obstacle (pillar) position
        self.obs_x = 380
        self.obs_y = 300
        self.obs_radius = 45
        
        # Camera projection matrices (Simulating two cameras at opposite corners)
        # Cam 1: Mounted high near Top-Left corner looking down at center
        # Cam 2: Mounted high near Bottom-Right corner looking down at center
        self.setup_virtual_cameras()

    def setup_virtual_cameras(self):
        """Sets up homography mappings from top-down arena to two corner camera views."""
        # 4 corners of arena in top-down space: TL, TR, BR, BL
        arena_corners = np.array([
            [0, 0],
            [self.arena_size, 0],
            [self.arena_size, self.arena_size],
            [0, self.arena_size]
        ], dtype=np.float32)

        # Camera 1 perspective (looking from top-left vantage point, 1280x720 frame)
        cam1_corners = np.array([
            [160, 90],    # TL looks closer
            [1050, 140],  # TR perspective compression
            [880, 640],   # BR
            [110, 580]    # BL
        ], dtype=np.float32)

        # Camera 2 perspective (looking from bottom-right vantage point, 1280x720 frame)
        cam2_corners = np.array([
            [400, 110],   # TL
            [1180, 180],  # TR
            [1120, 620],  # BR
            [220, 560]    # BL
        ], dtype=np.float32)

        # Perspective transforms: Top-Down Arena -> Raw Camera Frames
        self.H_arena_to_cam1, _ = cv2.findHomography(arena_corners, cam1_corners)
        self.H_arena_to_cam2, _ = cv2.findHomography(arena_corners, cam2_corners)
        
        # Ground truth calibration points for calibrator
        self.cam1_corners = cam1_corners.tolist()
        self.cam2_corners = cam2_corners.tolist()

    def update_simulation(self):
        """Updates physics and movements of both combat robots."""
        # Move Enemy Robot (bouncing around arena)
        self.enemy_x += self.enemy_vx
        self.enemy_y += self.enemy_vy
        
        margin = 80
        if self.enemy_x < margin or self.enemy_x > self.arena_size - margin:
            self.enemy_vx *= -1
        if self.enemy_y < margin or self.enemy_y > self.arena_size - margin:
            self.enemy_vy *= -1

        # Move My Robot towards Enemy Robot (simple simulated response)
        dx = self.enemy_x - self.my_x
        dy = self.enemy_y - self.my_y
        dist = math.hypot(dx, dy)
        target_angle = math.degrees(math.atan2(dy, dx))
        
        # Rotate heading towards enemy
        angle_diff = (target_angle - self.my_heading + 180.0) % 360.0 - 180.0
        self.my_heading += np.clip(angle_diff * 0.15, -6.0, 6.0)
        
        # Drive forward if facing approximately towards target
        if abs(angle_diff) < 30.0 and dist > 50:
            speed = 2.0
            rad = math.radians(self.my_heading)
            self.my_x += speed * math.cos(rad)
            self.my_y += speed * math.sin(rad)

    def render_arena(self) -> np.ndarray:
        """Renders the top-down ground-truth arena canvas."""
        # Solid white floor (combating conditions)
        arena = np.ones((self.arena_size, self.arena_size, 3), dtype=np.uint8) * 248
        
        # Arena boundary walls (dark steel rim)
        cv2.rectangle(arena, (0, 0), (self.arena_size, self.arena_size), (40, 40, 40), 12)
        
        # Arena hazard / corner post obstacle (causes Cam 1 occlusion)
        cv2.circle(arena, (self.obs_x, self.obs_y), self.obs_radius, (90, 90, 90), -1)
        cv2.circle(arena, (self.obs_x, self.obs_y), self.obs_radius, (30, 30, 30), 2)
        
        # 1. Render Enemy Robot (dark combat steel wedge / box)
        ex, ey = int(self.enemy_x), int(self.enemy_y)
        cv2.rectangle(arena, (ex - 24, ey - 24), (ex + 24, ey + 24), (35, 35, 45), -1)
        cv2.circle(arena, (ex, ey), 8, (180, 40, 40), -1)  # Red weapon spinner
        
        # 2. Render My Robot Body & ArUco Marker
        mx, my = int(self.my_x), int(self.my_y)
        # Robot base plate (Dark gray square)
        rot_mat = cv2.getRotationMatrix2D((mx, my), -self.my_heading - 90, 1.0)
        
        # Place ArUco marker on top plate
        th, tw, _ = self.tag_img.shape
        x1, y1 = mx - tw // 2, my - th // 2
        x2, y2 = x1 + tw, y1 + th
        
        if 0 <= x1 and x2 < self.arena_size and 0 <= y1 and y2 < self.arena_size:
            # Rotate tag according to heading
            tag_center = (tw // 2, th // 2)
            M_tag = cv2.getRotationMatrix2D(tag_center, -self.my_heading - 90, 1.0)
            rotated_tag = cv2.warpAffine(self.tag_img, M_tag, (tw, th), borderValue=(255, 255, 255))
            arena[y1:y2, x1:x2] = rotated_tag

        return arena

    def get_camera_frames(self):
        """Generates raw 1280x720 frames for Camera 1 and Camera 2 with realistic perspective."""
        self.update_simulation()
        arena = self.render_arena()
        
        # Warp top-down arena into Camera 1 perspective
        frame1 = cv2.warpPerspective(arena, self.H_arena_to_cam1, (1280, 720), borderValue=(20, 20, 20))
        
        # Warp top-down arena into Camera 2 perspective
        frame2 = cv2.warpPerspective(arena, self.H_arena_to_cam2, (1280, 720), borderValue=(20, 20, 20))
        
        # Simulate Cam 1 occlusion: if Enemy is behind the obstacle from Cam 1 vantage point
        # Draw a blindspot shadow or obstacle occlusion on Cam 1
        cam1_obs_pt = cv2.perspectiveTransform(
            np.array([[[self.obs_x, self.obs_y]]], dtype=np.float32),
            self.H_arena_to_cam1
        )[0][0]
        ox1, oy1 = int(cam1_obs_pt[0]), int(cam1_obs_pt[1])
        cv2.circle(frame1, (ox1, oy1), 38, (50, 50, 50), -1)
        cv2.putText(frame1, "ARENA PILLAR", (ox1 - 40, oy1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return frame1, frame2

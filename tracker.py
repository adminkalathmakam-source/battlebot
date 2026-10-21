"""
Bot Detection & Multi-Camera Tracking Fusion.
- Tracks My Robot using ArUco Marker (position + heading angle).
- Tracks Enemy Robot using floor contrast / blob detection against white arena floor.
- Dual camera switching & fallback logic when occlusion occurs.
"""

import math
import cv2
import numpy as np
from config import TrackingConfig, ArenaConfig

class BotTracker:
    def __init__(self, tracking_cfg: TrackingConfig, arena_cfg: ArenaConfig):
        self.cfg = tracking_cfg
        self.arena_cfg = arena_cfg
        
        # Setup ArUco Detector with OpenCV backward/forward compatibility
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.cfg.ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Check if ArucoDetector class exists (OpenCV >= 4.7.0)
        if hasattr(cv2.aruco, 'ArucoDetector'):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self.aruco_detector = None

    def detect_aruco_in_warped(self, warped_frame: np.ndarray):
        """
        Detects ArUco marker in top-down warped view.
        Returns: (x, y), heading_deg, tag_corners, detected_bool
        """
        if warped_frame is None:
            return None, None, None, False
            
        gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == self.cfg.MY_BOT_ARUCO_ID:
                    pts = corners[i][0]  # 4 corners: [TL, TR, BR, BL]
                    
                    # Centroid of the ArUco tag
                    cx = float(np.mean(pts[:, 0]))
                    cy = float(np.mean(pts[:, 1]))
                    
                    # Front edge midpoint (between corner 0 and corner 1)
                    front_x = (pts[0][0] + pts[1][0]) / 2.0
                    front_y = (pts[0][1] + pts[1][1]) / 2.0
                    
                    # Heading angle in degrees (-180 to +180)
                    dx = front_x - cx
                    dy = front_y - cy
                    heading_rad = math.atan2(dy, dx)
                    heading_deg = math.degrees(heading_rad)
                    
                    return (cx, cy), heading_deg, pts, True
                    
        return None, None, None, False

    def detect_enemy_blob(self, warped_frame: np.ndarray, my_bot_pos: tuple = None):
        """
        Detects Enemy Robot on the white arena floor using color contrast / background thresholding.
        Excludes 'my_bot_pos' region to prevent self-detection.
        Returns: (x, y), contour, area, detected_bool
        """
        if warped_frame is None:
            return None, None, 0, False
            
        hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        
        # Floor is solid white (High Value, Low Saturation)
        # Any object that is NOT white (dark metal, colored paint, wheels) is foreground
        # Condition for Non-White: Saturation > S_MAX OR Value < V_MIN
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Pixels that significantly deviate from white floor
        mask_sat = s_channel > self.cfg.FLOOR_WHITE_S_MAX
        mask_val = v_channel < self.cfg.FLOOR_WHITE_V_MIN
        bot_mask = (mask_sat | mask_val).astype(np.uint8) * 255
        
        # Ignore arena borders (e.g., 20 pixels from edges to eliminate wall shadows)
        border_margin = 25
        h, w = bot_mask.shape
        cv2.rectangle(bot_mask, (0, 0), (w, border_margin), 0, -1)
        cv2.rectangle(bot_mask, (0, h - border_margin), (w, h), 0, -1)
        cv2.rectangle(bot_mask, (0, 0), (border_margin, h), 0, -1)
        cv2.rectangle(bot_mask, (w - border_margin, 0), (w, h), 0, -1)
        
        # Mask out My Robot if currently located
        if my_bot_pos is not None:
            mx, my = int(my_bot_pos[0]), int(my_bot_pos[1])
            cv2.circle(bot_mask, (mx, my), self.cfg.MY_BOT_EXCLUSION_RADIUS, 0, -1)
            
        # Clean up noise (scuff marks, debris) with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bot_mask = cv2.morphologyEx(bot_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bot_mask = cv2.morphologyEx(bot_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(bot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_contour = None
        max_area = 0
        enemy_pos = None
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.cfg.ENEMY_MIN_AREA <= area <= self.cfg.ENEMY_MAX_AREA:
                if area > max_area:
                    max_area = area
                    best_contour = cnt
                    
        if best_contour is not None:
            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                return (cx, cy), best_contour, max_area, True
                
        return None, None, 0, False

    def process_and_fuse(self, warped1: np.ndarray, warped2: np.ndarray):
        """
        Processes warped frames from both cameras and applies camera switching / fallback logic.
        Returns a dict containing fused coordinates, headings, and camera source telemetry.
        """
        # 1. Track My Bot in Cam 1 & Cam 2
        my_pos1, heading1, pts1, found_my1 = self.detect_aruco_in_warped(warped1)
        my_pos2, heading2, pts2, found_my2 = self.detect_aruco_in_warped(warped2)
        
        fused_my_pos = None
        fused_heading = None
        my_bot_source = "NONE"
        
        if found_my1 and found_my2:
            # Both see my bot: use Cam 1 as primary or average them
            fused_my_pos = my_pos1
            fused_heading = heading1
            my_bot_source = "CAM1 (CAM2 CONFIRMED)"
        elif found_my1:
            fused_my_pos = my_pos1
            fused_heading = heading1
            my_bot_source = "CAM1"
        elif found_my2:
            # Fallback to Cam 2
            fused_my_pos = my_pos2
            fused_heading = heading2
            my_bot_source = "CAM2 (FALLBACK)"
            
        # 2. Track Enemy Bot in Cam 1 & Cam 2 (using fused_my_pos for exclusion)
        enemy_pos1, cnt1, area1, found_e1 = self.detect_enemy_blob(warped1, fused_my_pos)
        enemy_pos2, cnt2, area2, found_e2 = self.detect_enemy_blob(warped2, fused_my_pos)
        
        fused_enemy_pos = None
        enemy_source = "NONE"
        fused_enemy_area = 0
        
        if found_e1 and found_e2:
            # If both detect, select the one with clearer detection (larger unoccluded area) or Cam 1
            if area2 > area1 * 1.3:
                fused_enemy_pos = enemy_pos2
                enemy_source = "CAM2 (BETTER VIEW)"
                fused_enemy_area = area2
            else:
                fused_enemy_pos = enemy_pos1
                enemy_source = "CAM1 (CAM2 CONFIRMED)"
                fused_enemy_area = area1
        elif found_e1:
            fused_enemy_pos = enemy_pos1
            enemy_source = "CAM1"
            fused_enemy_area = area1
        elif found_e2:
            # Seamless fallback to Camera 2 when Cam 1 is occluded!
            fused_enemy_pos = enemy_pos2
            enemy_source = "CAM2 (FALLBACK)"
            fused_enemy_area = area2
            
        return {
            "my_bot": {
                "detected": fused_my_pos is not None,
                "pos": fused_my_pos,
                "heading": fused_heading,
                "source": my_bot_source,
                "cam1_detected": found_my1,
                "cam2_detected": found_my2
            },
            "enemy_bot": {
                "detected": fused_enemy_pos is not None,
                "pos": fused_enemy_pos,
                "area": fused_enemy_area,
                "source": enemy_source,
                "cam1_detected": found_e1,
                "cam2_detected": found_e2
            }
        }

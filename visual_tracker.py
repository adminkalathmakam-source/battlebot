"""
No-Tag Visual Photo Tracker & Floor Contrast Segmentation.
- Tracks Our Bot using an uploaded reference photo (with multi-angle rotation search for position & heading).
- Automatically tracks ANY other non-white object/blob on the white arena floor as the Enemy Bot!
- No enemy photo upload required.
"""

import os
import shutil
import time
import cv2
import numpy as np
from config import VisualTrackingConfig, ArenaConfig

class VisualRobotTracker:
    def __init__(self, track_cfg: VisualTrackingConfig, arena_cfg: ArenaConfig):
        self.cfg = track_cfg
        self.arena_cfg = arena_cfg
        
        self.my_template_bgr = None
        self.my_rotated_templates = []  # List of (angle, rotated_gray)
        
        # Enemy motion & orientation state
        self.enemy_pos_history = []
        self.enemy_smoothed_heading = 0.0
        self.enemy_has_heading = False
        
        template_path = getattr(self.cfg, 'MY_BOT_TEMPLATE_PATH', 'my_robot_template.png')
        self.load_my_template(template_path)


    def _build_rotated_templates(self, img_bgr: np.ndarray, target_size: int = 70):
        """Generates 360-degree rotated template pyramid from an image."""
        if img_bgr is None:
            return None, []
            
        h, w = img_bgr.shape[:2]
        scale = target_size / max(h, w)
        resized_bgr = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
        gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
        
        th, tw = gray.shape
        center = (tw // 2, th // 2)
        step = max(5, getattr(self.cfg, 'ROTATION_STEP_DEG', 10))
        
        rotated_list = []
        for angle in range(-180, 180, step):
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nw = int((th * sin) + (tw * cos))
            nh = int((th * cos) + (tw * sin))
            M[0, 2] += (nw / 2) - center[0]
            M[1, 2] += (nh / 2) - center[1]
            
            rotated = cv2.warpAffine(gray, M, (nw, nh), borderValue=255)
            rotated_list.append((angle, rotated))
            
        return resized_bgr, rotated_list

    def load_my_template(self, path: str = "my_robot_template.png") -> bool:
        """Loads reference photo for Our Bot and generates rotated templates."""
        if not os.path.exists(path):
            self.generate_default_template(path)
            
        img = cv2.imread(path)
        if img is not None:
            self.my_template_bgr, self.my_rotated_templates = self._build_rotated_templates(img)
            print(f"[OK] Our Bot template loaded from '{path}' ({len(self.my_rotated_templates)} rotation templates @ 10° steps)")
            return True
        return False

    def upload_my_template_file(self, src_file: str):
        """Replaces Our Bot reference template with a user-uploaded image file."""
        dest = getattr(self.cfg, 'MY_BOT_TEMPLATE_PATH', 'my_robot_template.png')
        shutil.copyfile(src_file, dest)
        self.load_my_template(dest)

    def set_my_template_from_roi(self, warped_frame: np.ndarray, roi: tuple):
        """Crops Our Bot directly from live camera feed."""
        x, y, w, h = roi
        if w > 15 and h > 15:
            crop = warped_frame[y:y+h, x:x+w]
            dest = getattr(self.cfg, 'MY_BOT_TEMPLATE_PATH', 'my_robot_template.png')
            cv2.imwrite(dest, crop)
            self.load_my_template(dest)

    def generate_default_template(self, save_path: str):
        """Generates a distinctive combat wedge template if none exists."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        pts = np.array([[50, 15], [85, 85], [15, 85]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (45, 45, 55))
        cv2.rectangle(img, (35, 30), (65, 70), (0, 180, 220), -1)
        cv2.arrowedLine(img, (50, 70), (50, 25), (0, 255, 0), 3, tipLength=0.3)
        cv2.imwrite(save_path, img)

    def track(self, warped_frame: np.ndarray):
        """
        Processes top-down arena view:
        1. Segments all non-white objects on the solid white floor.
        2. Matches Our Bot using the uploaded photo template (finds position + heading θ).
        3. Treats ANY other non-white object/blob on the arena floor as the Enemy Bot!
        """
        if warped_frame is None or not self.my_rotated_templates:
            return {
                "my_bot": {"detected": False, "pos": None, "heading": None, "score": 0.0},
                "enemy_bot": {"detected": False, "pos": None, "area": 0}
            }

        hsv = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Floor Background Segmentation (Solid White Arena Floor)
        # Any pixel with Saturation > S_MAX OR Value < V_MIN is foreground
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        mask_sat = s_channel > self.cfg.FLOOR_WHITE_S_MAX
        mask_val = v_channel < self.cfg.FLOOR_WHITE_V_MIN
        bot_mask = (mask_sat | mask_val).astype(np.uint8) * 255
        
        # Exclude arena boundary margins (eliminate wall shadows/seams)
        margin = 25
        h, w = bot_mask.shape
        cv2.rectangle(bot_mask, (0, 0), (w, margin), 0, -1)
        cv2.rectangle(bot_mask, (0, h - margin), (w, h), 0, -1)
        cv2.rectangle(bot_mask, (0, 0), (margin, h), 0, -1)
        cv2.rectangle(bot_mask, (w - margin, 0), (w, h), 0, -1)

        # Morphological noise cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bot_mask = cv2.morphologyEx(bot_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bot_mask = cv2.morphologyEx(bot_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(bot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.cfg.MIN_BOT_AREA <= area <= self.cfg.MAX_BOT_AREA:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    candidates.append({
                        "pos": (cx, cy),
                        "contour": cnt,
                        "area": area,
                        "bbox": (bx, by, bw, bh)
                    })

        my_bot = {"detected": False, "pos": None, "heading": None, "score": 0.0}
        enemy_bot = {"detected": False, "pos": None, "area": 0}

        if not candidates:
            return {"my_bot": my_bot, "enemy_bot": enemy_bot}

        # 2. Match Our Bot against the photo template with rotation search
        best_match_score = -1.0
        best_candidate_idx = -1
        best_heading = 0.0

        for idx, cand in enumerate(candidates):
            bx, by, bw, bh = cand["bbox"]
            pad = 20
            rx1 = max(0, bx - pad)
            ry1 = max(0, by - pad)
            rx2 = min(w, bx + bw + pad)
            ry2 = min(h, by + bh + pad)
            
            roi_gray = gray[ry1:ry2, rx1:rx2]
            rh, rw = roi_gray.shape
            
            for angle, rot_tmpl in self.my_rotated_templates:
                th, tw = rot_tmpl.shape
                if rw < tw or rh < th:
                    continue
                res = cv2.matchTemplate(roi_gray, rot_tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_match_score:
                    best_match_score = max_val
                    best_candidate_idx = idx
                    best_heading = angle

        if best_candidate_idx >= 0 and best_match_score >= self.cfg.MATCH_THRESHOLD:
            my_bot["detected"] = True
            my_bot["pos"] = candidates[best_candidate_idx]["pos"]
            my_bot["heading"] = best_heading
            my_bot["score"] = float(best_match_score)
            candidates.pop(best_candidate_idx)
        elif best_candidate_idx >= 0:
            # Fallback tracking if match score is moderate
            my_bot["detected"] = True
            my_bot["pos"] = candidates[best_candidate_idx]["pos"]
            my_bot["heading"] = best_heading
            my_bot["score"] = float(best_match_score)
            candidates.pop(best_candidate_idx)

        # 3. Enemy Bot = ANY OTHER non-white blob on the floor!
        # Find the largest/most prominent foreground blob outside Our Bot's radius
        best_enemy = None
        max_enemy_area = 0
        
        for cand in candidates:
            if my_bot["detected"]:
                dist = np.hypot(cand["pos"][0] - my_bot["pos"][0], cand["pos"][1] - my_bot["pos"][1])
                if dist < self.cfg.MY_BOT_EXCLUSION_RADIUS:
                    continue  # Ignore pieces or attachments of Our Bot
                    
            if cand["area"] > max_enemy_area:
                max_enemy_area = cand["area"]
                best_enemy = cand

        if best_enemy is not None:
            ecx, ecy = best_enemy["pos"]
            cnt = best_enemy["contour"]
            
            # Oriented Bounding Box
            rect = cv2.minAreaRect(cnt)
            (rcx, rcy), (rw, rh), angle = rect
            box_pts = cv2.boxPoints(rect)
            box_list = [[float(p[0]), float(p[1])] for p in box_pts]

            length = max(rw, rh)
            width = min(rw, rh)

            # Principal axis angle
            if rw >= rh:
                chassis_angle = float(angle)
            else:
                chassis_angle = float(angle + 90.0)
            chassis_angle = (chassis_angle + 180.0) % 360.0 - 180.0

            # Velocity vector tracking for forward heading resolution
            now = time.time()
            self.enemy_pos_history.append((now, ecx, ecy))
            # Keep history within last 0.5s
            self.enemy_pos_history = [p for p in self.enemy_pos_history if now - p[0] <= 0.6]

            if len(self.enemy_pos_history) >= 3:
                dx = self.enemy_pos_history[-1][1] - self.enemy_pos_history[0][1]
                dy = self.enemy_pos_history[-1][2] - self.enemy_pos_history[0][2]
                speed_px = float(np.hypot(dx, dy))

                if speed_px > 3.0:
                    vel_angle = float(np.degrees(np.arctan2(dy, dx)))
                    diff1 = abs((chassis_angle - vel_angle + 180.0) % 360.0 - 180.0)
                    diff2 = abs(((chassis_angle + 180.0) - vel_angle + 180.0) % 360.0 - 180.0)
                    resolved_heading = chassis_angle if diff1 <= diff2 else ((chassis_angle + 180.0 + 180.0) % 360.0 - 180.0)

                    # Smooth heading with EMA
                    if not self.enemy_has_heading:
                        self.enemy_smoothed_heading = resolved_heading
                        self.enemy_has_heading = True
                    else:
                        alpha = 0.28
                        h_rad = np.radians(self.enemy_smoothed_heading)
                        r_rad = np.radians(resolved_heading)
                        nx = (1.0 - alpha) * np.cos(h_rad) + alpha * np.cos(r_rad)
                        ny = (1.0 - alpha) * np.sin(h_rad) + alpha * np.sin(r_rad)
                        self.enemy_smoothed_heading = float(np.degrees(np.arctan2(ny, nx)))
                else:
                    if self.enemy_has_heading:
                        diff1 = abs((chassis_angle - self.enemy_smoothed_heading + 180.0) % 360.0 - 180.0)
                        diff2 = abs(((chassis_angle + 180.0) - self.enemy_smoothed_heading + 180.0) % 360.0 - 180.0)
                        self.enemy_smoothed_heading = chassis_angle if diff1 <= diff2 else ((chassis_angle + 180.0 + 180.0) % 360.0 - 180.0)
                    else:
                        self.enemy_smoothed_heading = chassis_angle
                        self.enemy_has_heading = True
            else:
                self.enemy_smoothed_heading = chassis_angle
                self.enemy_has_heading = True

            # Calculate Weak Spots & Danger Zone
            hdg_rad = np.radians(self.enemy_smoothed_heading)
            f_vec = (float(np.cos(hdg_rad)), float(np.sin(hdg_rad)))       # Front
            r_vec = (-f_vec[0], -f_vec[1])                                  # Rear (Weak spot #1)
            l_vec = (-f_vec[1], f_vec[0])                                   # Left Flank/Wheel (Weak spot #2)
            r_side_vec = (f_vec[1], -f_vec[0])                              # Right Flank/Wheel (Weak spot #3)

            d_rear = length * 0.5 + 26.0
            d_side = width * 0.5 + 22.0
            d_front = length * 0.5 + 22.0

            weak_rear = (round(ecx + d_rear * r_vec[0], 1), round(ecy + d_rear * r_vec[1], 1))
            weak_left = (round(ecx + d_side * l_vec[0], 1), round(ecy + d_side * l_vec[1], 1))
            weak_right = (round(ecx + d_side * r_side_vec[0], 1), round(ecy + d_side * r_side_vec[1], 1))
            front_weapon = (round(ecx + d_front * f_vec[0], 1), round(ecy + d_front * f_vec[1], 1))

            enemy_bot["detected"] = True
            enemy_bot["pos"] = (float(ecx), float(ecy))
            enemy_bot["area"] = int(best_enemy["area"])
            enemy_bot["heading"] = round(float(self.enemy_smoothed_heading), 1)
            enemy_bot["length"] = round(float(length), 1)
            enemy_bot["width"] = round(float(width), 1)
            enemy_bot["box_points"] = box_list
            enemy_bot["weak_spots"] = {
                "rear": weak_rear,
                "left_wheel": weak_left,
                "right_wheel": weak_right,
                "front_weapon": front_weapon
            }
            enemy_bot["danger_cone"] = {
                "center": (float(ecx), float(ecy)),
                "heading": round(float(self.enemy_smoothed_heading), 1),
                "angle_span": 65.0,
                "radius": 95.0
            }

        return {
            "my_bot": my_bot,
            "enemy_bot": enemy_bot
        }


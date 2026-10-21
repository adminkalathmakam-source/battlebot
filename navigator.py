"""
Autonomous Combat Tactics & Weak-Spot Steering Navigator.
Features:
- Enemy Weapon Danger-Zone Avoidance (curving around spinner/wedge)
- Weak-Spot Target Selection (Rear Blind-Spot, Left Wheel, Right Wheel, Smart Auto-Flank)
- Smooth Proportional Differential Steering with Flanking Pursuit Arcs
- 100% Throttle Kinetic Ramming upon weak-spot alignment
- Standardized ESP-NOW Serial Packets: <CMD,LeftPWM,RightPWM,Distance,HeadingError>\n
"""

import math
import numpy as np
from config import SteeringConfig, TacticsConfig

class Navigator:
    def __init__(self, steer_cfg: SteeringConfig = None, tactics_cfg: TacticsConfig = None):
        self.steer_cfg = steer_cfg if steer_cfg is not None else SteeringConfig()
        self.tactics_cfg = tactics_cfg if tactics_cfg is not None else TacticsConfig()
        self.current_tactic = getattr(self.tactics_cfg, 'TACTIC_MODE', 'SMART_FLANK')

    def set_tactic(self, tactic_mode: str):
        """Sets active combat tactic mode."""
        valid_modes = ["SMART_FLANK", "REAR_WEAK_SPOT", "LEFT_WHEEL", "RIGHT_WHEEL", "HEAD_ON"]
        if tactic_mode in valid_modes:
            self.current_tactic = tactic_mode

    @staticmethod
    def _normalize_angle(angle_deg: float) -> float:
        """Normalizes angle to [-180, +180] degrees."""
        return (angle_deg + 180.0) % 360.0 - 180.0

    def compute_steering(self, my_pos: tuple, my_heading_deg: float, enemy_data,
                         tactic_mode: str = None, user_throttle: int = None) -> dict:
        """
        Computes weak-spot target selection, weapon avoidance, pursuit trajectory,
        and continuous differential motor power for ESP-NOW transmission.
        Supports:
        - Full Autonomous: AI computes both throttle and steering
        - Co-Pilot Steer-by-Wire: User controls front/back throttle; AI controls left/right aim!
        """
        active_tactic = tactic_mode if tactic_mode is not None else self.current_tactic

        # Default fallback response if targets are missing
        if my_pos is None or my_heading_deg is None or enemy_data is None:
            return {
                "command": "SEARCH",
                "tactic_mode": active_tactic,
                "tactic_phase": "SEARCHING_ARENA",
                "target_name": "NONE",
                "target_pos": None,
                "in_danger_cone": False,
                "distance": 0.0,
                "distance_to_enemy": 0.0,
                "target_bearing": 0.0,
                "heading_error": 0.0,
                "throttle_left": 0,
                "throttle_right": 0,
                "flanking_waypoint": None,
                "packet": "<SEARCH,0,0,0,0>\n",
                "ai_tip": "Searching arena for combat robots... Make sure bot is on the white floor.",
                "ai_alert": "SEARCHING"
            }

        mx, my = float(my_pos[0]), float(my_pos[1])
        my_hdg = float(my_heading_deg)

        # Parse enemy data (support both dict or legacy tuple)
        if isinstance(enemy_data, dict):
            ex, ey = float(enemy_data["pos"][0]), float(enemy_data["pos"][1])
            enemy_hdg = enemy_data.get("heading")
            weak_spots = enemy_data.get("weak_spots", {})
        else:
            ex, ey = float(enemy_data[0]), float(enemy_data[1])
            enemy_hdg = None
            weak_spots = {}

        # Distance & Bearing from Our Bot to Enemy Bot Center
        dx_enemy = ex - mx
        dy_enemy = ey - my
        dist_enemy = float(math.hypot(dx_enemy, dy_enemy))
        bearing_to_enemy = float(math.degrees(math.atan2(dy_enemy, dx_enemy)))

        # 1. EVALUATE ENEMY WEAPON DANGER CONE
        dx_e_to_my = mx - ex
        dy_e_to_my = my - ey
        angle_e_to_my = float(math.degrees(math.atan2(dy_e_to_my, dx_e_to_my)))

        in_danger_cone = False
        danger_half_angle = getattr(self.tactics_cfg, 'WEAPON_DANGER_ANGLE_DEG', 65.0)
        danger_radius = getattr(self.tactics_cfg, 'WEAPON_DANGER_RADIUS', 95.0)

        if enemy_hdg is not None:
            rel_angle_from_enemy_front = self._normalize_angle(angle_e_to_my - enemy_hdg)
            if abs(rel_angle_from_enemy_front) <= danger_half_angle and dist_enemy <= (danger_radius * 1.35):
                in_danger_cone = True
        else:
            rel_angle_from_enemy_front = 0.0

        # 2. AUTONOMOUS WEAK-SPOT DETECTION & SELECTION
        # AI selects the most vulnerable weak spot automatically
        target_name = "ENEMY_BODY"
        target_pos = (ex, ey)

        if weak_spots and enemy_hdg is not None:
            # Automatic intelligent decision:
            # If behind enemy, strike rear directly
            if abs(rel_angle_from_enemy_front) >= 100.0:
                target_pos = weak_spots.get("rear", (ex, ey))
                target_name = "REAR_BLIND_SPOT"
            # If on left flank, target left wheel
            elif 35.0 <= rel_angle_from_enemy_front < 100.0:
                target_pos = weak_spots.get("left_wheel", (ex, ey))
                target_name = "LEFT_WHEEL"
            # If on right flank, target right wheel
            elif -100.0 < rel_angle_from_enemy_front <= -35.0:
                target_pos = weak_spots.get("right_wheel", (ex, ey))
                target_name = "RIGHT_WHEEL"
            else:
                # Facing the front weapon: select closest flank to escape and hit wheel
                if rel_angle_from_enemy_front >= 0:
                    target_pos = weak_spots.get("left_wheel", (ex, ey))
                    target_name = "FLANK_L_WHEEL"
                else:
                    target_pos = weak_spots.get("right_wheel", (ex, ey))
                    target_name = "FLANK_R_WHEEL"

        # 3. WEAPON AVOIDANCE & FLANKING WAYPOINT
        flanking_waypoint = None
        steer_target_pos = target_pos
        tactic_phase = "PURSUIT"

        if in_danger_cone:
            orbit_rad = getattr(self.tactics_cfg, 'FLANKING_ORBIT_RADIUS', 115.0)
            escape_sign = 1.0 if rel_angle_from_enemy_front >= 0 else -1.0
            flank_bearing = math.radians(enemy_hdg + escape_sign * 105.0)

            wx = round(ex + orbit_rad * math.cos(flank_bearing), 1)
            wy = round(ey + orbit_rad * math.sin(flank_bearing), 1)
            flanking_waypoint = (wx, wy)
            steer_target_pos = (wx, wy)
            tactic_phase = "FLANKING_WEAPON"
        else:
            tactic_phase = "LOCK_ON_WEAK_SPOT"

        # 4. STEERING ERROR TO ACTIVE TARGET
        tx, ty = steer_target_pos
        dx_target = tx - mx
        dy_target = ty - my
        dist_to_target = float(math.hypot(dx_target, dy_target))
        bearing_to_target = float(math.degrees(math.atan2(dy_target, dx_target)))

        heading_error = self._normalize_angle(bearing_to_target - my_hdg)

        # 5. MOTOR POWER CALCULATION (USER THROTTLE VS FULL AUTO)
        tol = getattr(self.steer_cfg, 'ANGLE_TOLERANCE_DEG', 18.0)
        prox = getattr(self.steer_cfg, 'ATTACK_PROXIMITY_PIXELS', 75.0)

        max_pwr = getattr(self.steer_cfg, 'MAX_THROTTLE', 240)
        ram_pwr = getattr(self.steer_cfg, 'RAM_THROTTLE', 255)
        cruise_pwr = getattr(self.steer_cfg, 'CRUISE_THROTTLE', 190)
        turn_pwr = getattr(self.steer_cfg, 'TURN_THROTTLE', 165)

        kp = 1.35
        effort = kp * heading_error

        is_ram_ready = (dist_enemy <= prox or dist_to_target <= (prox * 0.9)) and abs(heading_error) <= (tol * 1.4)

        if user_throttle is not None:
            # === CO-PILOT STEER-BY-WIRE MODE ===
            # User controls Throttle (Forward/Back), AI controls Left/Right aim!
            if abs(user_throttle) > 15:
                # User driving forward or reverse
                left_pwm = int(np.clip(user_throttle - effort, -255, 255))
                right_pwm = int(np.clip(user_throttle + effort, -255, 255))
                command = "CO_FWD" if user_throttle > 0 else "CO_REV"
            else:
                # User throttle neutral: gently pivot to keep target dead-center
                if abs(heading_error) > tol:
                    piv = int(np.clip(effort * 0.7, -turn_pwr, turn_pwr))
                    left_pwm = -piv
                    right_pwm = piv
                    command = "AIM_LOCK"
                else:
                    left_pwm = 0
                    right_pwm = 0
                    command = "AIM_READY"
        else:
            # === FULL AUTONOMOUS MODE ===
            if is_ram_ready and not in_danger_cone:
                command = "RAM_WEAK"
                tactic_phase = "IMPACT_RAM"
                left_pwm = ram_pwr
                right_pwm = ram_pwr
            else:
                if abs(heading_error) <= tol:
                    command = "PURSUIT"
                    left_pwm = int(np.clip(max_pwr - effort, 0, 255))
                    right_pwm = int(np.clip(max_pwr + effort, 0, 255))
                elif abs(heading_error) <= 85.0:
                    command = "FLANK_ARC"
                    left_pwm = int(np.clip(cruise_pwr - 1.6 * effort, -180, 255))
                    right_pwm = int(np.clip(cruise_pwr + 1.6 * effort, -180, 255))
                else:
                    if heading_error > 0:
                        command = "SPIN_RIGHT"
                        left_pwm = turn_pwr
                        right_pwm = -turn_pwr
                    else:
                        command = "SPIN_LEFT"
                        left_pwm = -turn_pwr
                        right_pwm = turn_pwr

        # 6. AI DRIVING COACH & TACTICAL ADVISOR
        if in_danger_cone:
            ai_tip = "⚠️ DANGER: Enemy weapon spinner is facing us! Flanking to safety — hold back [S]!"
            ai_alert = "DANGER"
        elif is_ram_ready:
            ai_tip = f"💥 WEAK SPOT IN RANGE! Press [W] FULL THROTTLE to smash their {target_name}!"
            ai_alert = "RAM_NOW"
        elif target_name == "REAR_BLIND_SPOT":
            ai_tip = "🎯 LOCKED ON REAR MOTORS: Gun throttle forward [W] — AI is holding your crosshair locked on their back!"
            ai_alert = "ATTACK"
        elif "WHEEL" in target_name:
            ai_tip = f"⚡ EXPOSED {target_name}: Drive forward [W] to take out their drive wheel for an instant TKO!"
            ai_alert = "ATTACK"
        elif abs(heading_error) > 45:
            ai_tip = "🔄 AI is pivoting left/right to lock onto enemy weak spot..."
            ai_alert = "ALIGNING"
        else:
            ai_tip = f"✅ Target Locked ({target_name}). Press [W] forward to strike."
            ai_alert = "LOCKED"

        cmd_short = command[:10]
        packet = f"<{cmd_short},{int(left_pwm)},{int(right_pwm)},{int(dist_enemy)},{int(heading_error)}>\n"

        return {
            "command": command,
            "tactic_mode": active_tactic,
            "tactic_phase": tactic_phase,
            "target_name": target_name,
            "target_pos": target_pos,
            "steer_target_pos": steer_target_pos,
            "in_danger_cone": in_danger_cone,
            "distance": round(dist_to_target, 1),
            "distance_to_enemy": round(dist_enemy, 1),
            "target_bearing": round(bearing_to_target, 1),
            "heading_error": round(heading_error, 1),
            "throttle_left": int(left_pwm),
            "throttle_right": int(right_pwm),
            "flanking_waypoint": flanking_waypoint,
            "packet": packet,
            "ai_tip": ai_tip,
            "ai_alert": ai_alert
        }


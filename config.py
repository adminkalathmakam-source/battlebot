"""
Configuration settings for Distributed Dual-Laptop Combat Arena Tracking System.
Supports No-Tag Visual Photo Template Matching, UDP Network Coordinate Stream,
and ESP-NOW Serial Transmission on Master Laptop.
"""

from dataclasses import dataclass
import os

@dataclass
class NetworkConfig:
    # IP Address of Master Laptop (ASUS TUF 1).
    # Secondary Laptop sends UDP packets to this IP.
    # When testing on a single laptop, use "127.0.0.1".
    # In arena: Connect both laptops to same Wi-Fi/Hotspot and set to Master's local IP (e.g., "192.168.1.100")
    MASTER_IP: str = "127.0.0.1"
    UDP_PORT: int = 5005
    BUFFER_SIZE: int = 2048
    SOCKET_TIMEOUT: float = 0.05

@dataclass
class ArenaConfig:
    # Standardized top-down warped 2D arena floor dimensions (pixels)
    # Both laptops warp their perspectives into this EXACT SAME coordinate space (0 to 800)
    WARP_WIDTH: int = 800
    WARP_HEIGHT: int = 800

@dataclass
class CameraConfig:
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    FPS: int = 30

@dataclass
class VisualTrackingConfig:
    # Path to reference photo/template of our robot
    TEMPLATE_PATH: str = "my_robot_template.png"
    MY_BOT_TEMPLATE_PATH: str = "my_robot_template.png"
    
    # Path to reference photo/template of enemy robot (optional, falls back to contrast blob)
    ENEMY_BOT_TEMPLATE_PATH: str = "enemy_robot_template.png"
    
    # Template matching threshold (0.0 to 1.0)
    MATCH_THRESHOLD: float = 0.50
    
    # Rotation angle step for multi-angle template search (degrees)
    # 10 deg provides high precision with low CPU overhead (~36 angles)
    ROTATION_STEP_DEG: int = 10
    
    # Floor detection thresholds (solid white arena floor)
    # Non-white pixels (metal, colored paint, tires, weapons) are foreground
    FLOOR_WHITE_V_MIN: int = 180     # Brightness threshold for white floor
    FLOOR_WHITE_S_MAX: int = 60      # Saturation threshold for white floor
    
    # Robot blob size filters in 800x800 arena space
    MIN_BOT_AREA: int = 500
    MAX_BOT_AREA: int = 35000
    
    # Exclusion radius around My Bot when locating Enemy Bot
    MY_BOT_EXCLUSION_RADIUS: int = 95  # pixels

@dataclass
class TacticsConfig:
    # Selected combat tactic:
    # "SMART_FLANK" (Default: curves around weapon to strike nearest weak spot)
    # "REAR_WEAK_SPOT" (Target blind-spot / motor drive at back)
    # "LEFT_WHEEL"  (Target exposed left wheel)
    # "RIGHT_WHEEL" (Target exposed right wheel)
    # "HEAD_ON"     (Direct frontal collision for wedge/pushers)
    TACTIC_MODE: str = "SMART_FLANK"
    
    # Distance in pixels behind/beside enemy to place the weak spot aim point
    WEAK_SPOT_STANDOFF: float = 35.0
    
    # Weapon danger zone parameters (Front of enemy bot)
    WEAPON_DANGER_RADIUS: float = 90.0      # Pixels in front of enemy
    WEAPON_DANGER_ANGLE_DEG: float = 65.0   # Half-angle of front weapon cone (±65°)
    FLANKING_ORBIT_RADIUS: float = 115.0    # Waypoint distance when flanking around weapon

@dataclass
class SteeringConfig:
    ANGLE_TOLERANCE_DEG: float = 18.0
    ATTACK_PROXIMITY_PIXELS: float = 75.0
    MAX_THROTTLE: int = 240
    CRUISE_THROTTLE: int = 190
    TURN_THROTTLE: int = 165
    RAM_THROTTLE: int = 255

@dataclass
class SerialConfig:
    # Only enabled on MASTER laptop where ESP32 transmitter is plugged in
    ENABLED: bool = True
    PORT: str = "COM3"
    BAUD_RATE: int = 115200
    TIMEOUT: float = 0.05
    SEND_INTERVAL_SEC: float = 0.033


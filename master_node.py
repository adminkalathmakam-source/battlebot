"""
Master Laptop Node (ASUS TUF 1 - Corner Camera A + Serial ESP-NOW Transmitter).
Runs local visual photo-template tracking, receives coordinates from Laptop 2 over UDP,
fuses dual-corner streams with automatic occlusion fallback, calculates steering vectors,
and transmits commands to the ESP-NOW transmitter via PySerial.
"""

import sys
import time
import argparse
import cv2
import numpy as np

from config import ArenaConfig, CameraConfig, VisualTrackingConfig, NetworkConfig, SteeringConfig, SerialConfig
from camera_stream import CameraStream
from homography import ArenaCalibrator
from visual_tracker import VisualRobotTracker
from navigator import Navigator
from serial_bridge import SerialBridge
from network_bridge import UDPCornerReceiver

def parse_args():
    parser = argparse.ArgumentParser(description="Master Laptop Node (Laptop 1)")
    parser.add_argument("--cam-idx", type=int, default=0,
                        help="Webcam device index on this laptop (default 0)")
    parser.add_argument("--serial-port", type=str, default=None,
                        help="COM port of ESP-NOW transmitter (overrides config.py)")
    parser.add_argument("--calib-file", type=str, default="calibration_master.json",
                        help="Calibration JSON filename for this corner")
    return parser.parse_args()

def fuse_telemetry(local_track: dict, remote_telemetry: dict, remote_is_fresh: bool):
    """
    Fuses local Camera 1 tracking data with remote Laptop 2 UDP stream.
    Applies seamless fallback if local camera loses sight or is occluded.
    """
    my_local = local_track["my_bot"]
    enemy_local = local_track["enemy_bot"]

    my_remote = remote_telemetry.get("my_bot", {}) if (remote_is_fresh and remote_telemetry) else {}
    enemy_remote = remote_telemetry.get("enemy_bot", {}) if (remote_is_fresh and remote_telemetry) else {}

    # 1. My Robot Fusion
    fused_my = {"detected": False, "pos": None, "heading": None, "source": "NONE"}
    
    if my_local["detected"]:
        fused_my["detected"] = True
        fused_my["pos"] = my_local["pos"]
        fused_my["heading"] = my_local["heading"]
        if my_remote.get("detected"):
            fused_my["source"] = "CAM1 (CAM2 CONFIRMED)"
        else:
            fused_my["source"] = "CAM1 (LOCAL)"
    elif my_remote.get("detected") and my_remote.get("x") is not None:
        # Fallback to Laptop 2
        fused_my["detected"] = True
        fused_my["pos"] = (my_remote["x"], my_remote["y"])
        fused_my["heading"] = my_remote.get("heading")
        fused_my["source"] = "LAPTOP 2 (FALLBACK)"

    # 2. Enemy Robot Fusion
    fused_enemy = {"detected": False, "pos": None, "source": "NONE"}
    
    if enemy_local["detected"]:
        fused_enemy["detected"] = True
        fused_enemy["pos"] = enemy_local["pos"]
        if enemy_remote.get("detected"):
            fused_enemy["source"] = "CAM1 (CAM2 CONFIRMED)"
        else:
            fused_enemy["source"] = "CAM1 (LOCAL)"
    elif enemy_remote.get("detected") and enemy_remote.get("x") is not None:
        # Fallback to Laptop 2 when Cam 1 is occluded
        fused_enemy["detected"] = True
        fused_enemy["pos"] = (enemy_remote["x"], enemy_remote["y"])
        fused_enemy["source"] = "LAPTOP 2 (OCCLUSION FALLBACK)"

    return fused_my, fused_enemy

def draw_master_hud(canvas: np.ndarray, my_bot: dict, enemy_bot: dict, 
                    steering: dict, remote_fresh: bool, serial_stat: str, fps: float):
    """Draws master tactical HUD and steering vectors."""
    h, w = canvas.shape[:2]
    cmd = steering["command"]
    dist = steering["distance"]
    err = steering["heading_error"]

    # 1. My Robot (Green)
    if my_bot["detected"]:
        mx, my = int(my_bot["pos"][0]), int(my_bot["pos"][1])
        hdg = my_bot.get("heading")
        cv2.circle(canvas, (mx, my), 28, (0, 255, 0), 2)
        cv2.circle(canvas, (mx, my), 4, (0, 255, 255), -1)
        if hdg is not None:
            rad = np.radians(hdg)
            ax = int(mx + 45 * np.cos(rad))
            ay = int(my + 45 * np.sin(rad))
            cv2.arrowedLine(canvas, (mx, my), (ax, ay), (0, 255, 255), 3, tipLength=0.3)
        cv2.putText(canvas, f"MY BOT [{my_bot['source']}]", (mx - 40, my - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # 2. Enemy Robot (Red)
    if enemy_bot["detected"]:
        ex, ey = int(enemy_bot["pos"][0]), int(enemy_bot["pos"][1])
        cv2.circle(canvas, (ex, ey), 30, (0, 0, 255), 2)
        cv2.circle(canvas, (ex, ey), 5, (0, 0, 255), -1)
        cv2.line(canvas, (ex - 35, ey), (ex + 35, ey), (0, 0, 255), 1)
        cv2.line(canvas, (ex, ey - 35), (ex, ey + 35), (0, 0, 255), 1)
        cv2.putText(canvas, f"ENEMY [{enemy_bot['source']}]", (ex - 40, ey - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # 3. Target Vector Line
    if my_bot["detected"] and enemy_bot["detected"]:
        mx, my = int(my_bot["pos"][0]), int(my_bot["pos"][1])
        ex, ey = int(enemy_bot["pos"][0]), int(enemy_bot["pos"][1])
        color = (0, 255, 0) if abs(err) <= 20 else ((0, 255, 255) if abs(err) <= 60 else (0, 165, 255))
        cv2.line(canvas, (mx, my), (ex, ey), color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{int(dist)}px", ((mx + ex) // 2 + 8, (my + ey) // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Top Status Banner
    cv2.rectangle(canvas, (0, 0), (w, 75), (20, 20, 25), -1)
    cmd_col = (0, 255, 0) if cmd == "FORWARD" else ((0, 0, 255) if cmd in ["RAM", "STOP"] else (255, 200, 0))
    cv2.putText(canvas, f"CMD: {cmd}", (16, 30), cv2.FONT_HERSHEY_DUPLEX, 0.85, cmd_col, 2)
    cv2.putText(canvas, f"ERR: {err:+.1f} deg | DIST: {int(dist)}px", (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    
    cv2.putText(canvas, f"FPS: {fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    net_col = (0, 255, 0) if remote_fresh else (0, 0, 255)
    cv2.putText(canvas, f"LAPTOP 2 UDP: {'CONNECTED' if remote_fresh else 'WAITING'}", 
                (w - 240, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, net_col, 1)

def main():
    args = parse_args()

    print("=" * 65)
    print("   MASTER LAPTOP NODE - CORNER A + UDP RECEIVER + ESP-NOW   ")
    print("=" * 65)

    arena_cfg = ArenaConfig()
    cam_cfg = CameraConfig()
    track_cfg = VisualTrackingConfig()
    steer_cfg = SteeringConfig()
    serial_cfg = SerialConfig()
    net_cfg = NetworkConfig()

    if args.serial_port:
        serial_cfg.PORT = args.serial_port

    calibrator = ArenaCalibrator(arena_cfg.WARP_WIDTH, arena_cfg.WARP_HEIGHT, calib_file=args.calib_file)
    
    # Initialize Camera Stream
    print(f"\n[INFO] Connecting to Master Webcam on index {args.cam_idx}...")
    cam = CameraStream(args.cam_idx, cam_cfg.FRAME_WIDTH, cam_cfg.FRAME_HEIGHT, cam_cfg.FPS, "Master-Cam")
    time.sleep(0.5)

    if not cam.is_opened:
        print(f"[ERROR] Could not open webcam at index {args.cam_idx}. Exiting.")
        return

    # Visual template tracker (No ArUco tags)
    tracker = VisualRobotTracker(track_cfg, arena_cfg)
    navigator = Navigator(steer_cfg)
    serial_bridge = SerialBridge(serial_cfg)
    
    # UDP Receiver for Laptop 2
    udp_receiver = UDPCornerReceiver(net_cfg)

    print(f"[OK] Master node active on port {net_cfg.UDP_PORT}. Waiting for Laptop 2...")
    print("     Controls: 'c': Calibrate 4 Corners | 't': Crop robot template ROI | 'q': Exit")

    fps_time = time.time()
    frames = 0
    current_fps = 0.0

    estop_active = False
    trigger_calib = False
    trigger_crop = False
    trigger_reset = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frames += 1
            if time.time() - fps_time >= 1.0:
                current_fps = frames / (time.time() - fps_time)
                frames = 0
                fps_time = time.time()

            # Default calibration setup if uncalibrated
            if calibrator.H_cam1 is None:
                calibrator.set_default_camera_corners(1, frame.shape[1], frame.shape[0])

            # Perspective warp to standardized 800x800 arena floor
            warped = calibrator.warp(frame, cam_num=1)

            # 1. Local Visual Tracking (Camera 1)
            local_track = tracker.track(warped)

            # 2. Receive Fresh Telemetry from Laptop 2 via UDP
            remote_telemetry, remote_fresh = udp_receiver.get_latest_telemetry()

            # 3. Dual-Laptop Fusion & Fallback Logic
            fused_my, fused_enemy = fuse_telemetry(local_track, remote_telemetry, remote_fresh)

            # 4. Target Vector & Steering Calculation
            steering = navigator.compute_steering(
                my_pos=fused_my["pos"],
                my_heading_deg=fused_my["heading"],
                enemy_pos=fused_enemy["pos"]
            )

            # 5. Serial Transmission to ESP-NOW Master Transmitter
            if estop_active:
                pkt = "<STOP,0,0,0,0>\n"
            else:
                pkt = steering["packet"]

            if serial_cfg.ENABLED and serial_bridge.is_connected:
                serial_bridge.send_packet(pkt)
                serial_stat = "E-STOP" if estop_active else "ONLINE"
            else:
                serial_stat = "E-STOP" if estop_active else "MOCK / READY"

            # Print telemetry summary periodically
            if frames % 30 == 0:
                print(f"[MASTER] MyBot: {fused_my['source']} | Enemy: {fused_enemy['source']} | Cmd: {steering['command']} | Packet: {steering['packet'].strip()}")

            # 6. Render Tactical Dashboard & Clickable Buttons
            canvas = warped.copy()
            draw_master_hud(canvas, fused_my, fused_enemy, steering, remote_fresh, serial_stat, current_fps)

            # Combine Warped View with Raw Master Cam thumbnail
            raw_thumb = cv2.resize(frame, (280, int(280 * frame.shape[0] / frame.shape[1])))
            cv2.putText(raw_thumb, "RAW CORNER A", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            combined = np.hstack([canvas, cv2.resize(raw_thumb, (280, canvas.shape[0]))])

            # Draw On-Screen Clickable Buttons Bar at bottom
            ch, cw = combined.shape[:2]
            cv2.rectangle(combined, (0, ch - 50), (cw, ch), (15, 15, 22), -1)

            # Button 1: E-STOP
            stop_col = (0, 0, 255) if not estop_active else (0, 165, 255)
            stop_txt = "STOP / E-STOP" if not estop_active else "RESUME STEERING"
            cv2.rectangle(combined, (12, ch - 44), (190, ch - 8), stop_col, -1)
            cv2.putText(combined, stop_txt, (25, ch - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

            # Button 2: CALIBRATE
            cv2.rectangle(combined, (200, ch - 44), (370, ch - 8), (75, 75, 125), -1)
            cv2.putText(combined, "CALIBRATE CORNERS", (210, ch - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

            # Button 3: CROP TEMPLATE
            cv2.rectangle(combined, (380, ch - 44), (550, ch - 8), (40, 110, 80), -1)
            cv2.putText(combined, "CROP TEMPLATE", (398, ch - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

            # Button 4: RESET
            cv2.rectangle(combined, (560, ch - 44), (700, ch - 8), (55, 55, 65), -1)
            cv2.putText(combined, "RESET ARENA", (580, ch - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            window_name = "Master Node (Corner A + ESP-NOW Link) - Tactical Arena Ground Truth"
            cv2.imshow(window_name, combined)

            # Setup mouse click handler for buttons
            def on_mouse_click(event, x, y, flags, param):
                nonlocal estop_active, trigger_calib, trigger_crop, trigger_reset
                if event == cv2.EVENT_LBUTTONDOWN and y >= ch - 50:
                    if 12 <= x <= 190:
                        estop_active = not estop_active
                        print(f"\n[ACTION] E-STOP Toggled: {'ACTIVE (STOPPING MOTORS)' if estop_active else 'RESUMED'}")
                    elif 200 <= x <= 370:
                        trigger_calib = True
                    elif 380 <= x <= 550:
                        trigger_crop = True
                    elif 560 <= x <= 700:
                        trigger_reset = True

            cv2.setMouseCallback(window_name, on_mouse_click)

            # Check button triggers or keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), 27]:
                break
            elif key == ord('c') or trigger_calib:
                trigger_calib = False
                calibrator.calibrate_camera(1, frame)
            elif key == ord('t') or trigger_crop:
                trigger_crop = False
                roi = cv2.selectROI("Crop My Robot Reference Template", warped, showCrosshair=True)
                cv2.destroyWindow("Crop My Robot Reference Template")
                if roi[2] > 0 and roi[3] > 0:
                    tracker.set_template_from_roi(warped, roi)
            elif key == ord('r') or trigger_reset:
                trigger_reset = False
                calibrator.H_cam1 = None
                print("[INFO] Arena calibration reset to default.")

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Shutting down master node...")
        serial_bridge.close()
        udp_receiver.close()
        cam.stop()
        cv2.destroyAllWindows()
        print("[OK] Master node stopped.")

if __name__ == "__main__":
    main()

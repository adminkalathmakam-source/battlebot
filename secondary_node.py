"""
Secondary Laptop Node (ASUS TUF 2 - Corner Camera B).
Runs visual photo-template tracking on local corner webcam, warps perspective into
standardized arena coordinates, and streams tracked coordinates over UDP to the Master Laptop.
"""

import sys
import time
import argparse
import cv2
import numpy as np

from config import ArenaConfig, CameraConfig, VisualTrackingConfig, NetworkConfig
from camera_stream import CameraStream
from homography import ArenaCalibrator
from visual_tracker import VisualRobotTracker
from network_bridge import UDPCornerSender

def parse_args():
    parser = argparse.ArgumentParser(description="Secondary Corner Camera Node (Laptop 2)")
    parser.add_argument("--master-ip", type=str, default=None, 
                        help="IP address of Master Laptop (overrides config.py)")
    parser.add_argument("--cam-idx", type=int, default=0,
                        help="Webcam device index on this laptop (default 0)")
    parser.add_argument("--calib-file", type=str, default="calibration_secondary.json",
                        help="Calibration JSON filename for this corner")
    return parser.parse_args()

def draw_node_hud(display: np.ndarray, tracking: dict, dest_ip: str, fps: float):
    """Draws visual tracking HUD on secondary laptop monitor."""
    h, w = display.shape[:2]
    my_bot = tracking["my_bot"]
    enemy_bot = tracking["enemy_bot"]
    
    # My Bot (Green)
    if my_bot["detected"]:
        mx, my = int(my_bot["pos"][0]), int(my_bot["pos"][1])
        hdg = my_bot["heading"]
        cv2.circle(display, (mx, my), 28, (0, 255, 0), 2)
        cv2.circle(display, (mx, my), 4, (0, 255, 255), -1)
        rad = np.radians(hdg)
        ax = int(mx + 45 * np.cos(rad))
        ay = int(my + 45 * np.sin(rad))
        cv2.arrowedLine(display, (mx, my), (ax, ay), (0, 255, 255), 3, tipLength=0.3)
        cv2.putText(display, f"MY BOT ({my_bot['score']:.2f})", (mx - 40, my - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Enemy Bot (Red)
    if enemy_bot["detected"]:
        ex, ey = int(enemy_bot["pos"][0]), int(enemy_bot["pos"][1])
        cv2.circle(display, (ex, ey), 30, (0, 0, 255), 2)
        cv2.circle(display, (ex, ey), 5, (0, 0, 255), -1)
        cv2.putText(display, "ENEMY", (ex - 25, ey - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # Header Bar
    cv2.rectangle(display, (0, 0), (w, 50), (20, 20, 25), -1)
    cv2.putText(display, f"SECONDARY NODE (CORNER B) -> UDP STREAM TO {dest_ip}", 
                (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
    
    status_str = f"FPS: {fps:.1f} | MyBot: {'LOCKED' if my_bot['detected'] else 'SEARCH'} | Enemy: {'LOCKED' if enemy_bot['detected'] else 'SEARCH'}"
    cv2.putText(display, status_str, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

def main():
    args = parse_args()
    
    print("=" * 65)
    print("   SECONDARY LAPTOP NODE - CORNER CAMERA UDP STREAMER   ")
    print("=" * 65)
    
    net_cfg = NetworkConfig()
    if args.master_ip:
        net_cfg.MASTER_IP = args.master_ip
        
    arena_cfg = ArenaConfig()
    cam_cfg = CameraConfig()
    track_cfg = VisualTrackingConfig()
    
    # Initialize homography calibrator for Secondary Laptop
    calibrator = ArenaCalibrator(arena_cfg.WARP_WIDTH, arena_cfg.WARP_HEIGHT, calib_file=args.calib_file)
    
    # Initialize Camera Stream with DirectShow
    print(f"\n[INFO] Connecting to webcam on index {args.cam_idx}...")
    cam = CameraStream(args.cam_idx, cam_cfg.FRAME_WIDTH, cam_cfg.FRAME_HEIGHT, cam_cfg.FPS, "Secondary-Cam")
    time.sleep(0.5)
    
    if not cam.is_opened:
        print(f"[ERROR] Could not open webcam at index {args.cam_idx}. Exiting.")
        return

    # Visual template tracker (No ArUco tags)
    tracker = VisualRobotTracker(track_cfg, arena_cfg)
    
    # UDP Network Sender
    sender = UDPCornerSender(net_cfg)
    
    print(f"[OK] Secondary node streaming coordinates to Master at {net_cfg.MASTER_IP}:{net_cfg.UDP_PORT}")
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
            
            # Track bots visually
            tracking = tracker.track(warped)
            
            # Broadcast coordinates over UDP to Master Laptop (including E-STOP status)
            sender.send_telemetry(tracking["my_bot"], tracking["enemy_bot"], cam_id=2, estop=estop_active)
            
            # Render local monitor
            disp = warped.copy()
            draw_node_hud(disp, tracking, net_cfg.MASTER_IP, current_fps)
            
            # Show live composite: Warped + Raw Corner
            raw_thumb = cv2.resize(frame, (280, int(280 * frame.shape[0] / frame.shape[1])))
            cv2.putText(raw_thumb, "RAW CORNER B", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            combined = np.hstack([disp, cv2.resize(raw_thumb, (280, disp.shape[0]))])

            # Draw On-Screen Clickable Buttons Bar at bottom
            ch, cw = combined.shape[:2]
            cv2.rectangle(combined, (0, ch - 50), (cw, ch), (15, 15, 22), -1)

            # Button 1: E-STOP
            stop_col = (0, 0, 255) if not estop_active else (0, 165, 255)
            stop_txt = "STOP / E-STOP" if not estop_active else "RESUME"
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

            window_name = "Secondary Node (Corner B) - Arena Telemetry Monitor"
            cv2.imshow(window_name, combined)

            def on_mouse_click(event, x, y, flags, param):
                nonlocal estop_active, trigger_calib, trigger_crop, trigger_reset
                if event == cv2.EVENT_LBUTTONDOWN and y >= ch - 50:
                    if 12 <= x <= 190:
                        estop_active = not estop_active
                        print(f"\n[ACTION] E-STOP Toggled on Secondary Node: {'STOPPED' if estop_active else 'RESUMED'}")
                    elif 200 <= x <= 370:
                        trigger_calib = True
                    elif 380 <= x <= 550:
                        trigger_crop = True
                    elif 560 <= x <= 700:
                        trigger_reset = True

            cv2.setMouseCallback(window_name, on_mouse_click)

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
                print("[INFO] Arena bounds reset to default.")

    except KeyboardInterrupt:
        pass
    finally:
        print("\n[INFO] Shutting down secondary node...")
        sender.close()
        cam.stop()
        cv2.destroyAllWindows()
        print("[OK] Secondary node stopped.")

if __name__ == "__main__":
    main()

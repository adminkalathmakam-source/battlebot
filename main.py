"""
Main Combat Arena Robot Tracker & Steering System.
Integrates Dual-Camera Streams (or Virtual Simulator), Homography Warping,
Bot Detection, Target Vector Math, Live Tactical HUD, and Serial Transmission to ESP32 ESP-NOW Bridge.
"""

import sys
import time
import argparse
import cv2
import numpy as np

from config import ArenaConfig, CameraConfig, TrackingConfig, SteeringConfig, SerialConfig
from camera_stream import CameraStream
from homography import ArenaCalibrator
from tracker import BotTracker
from navigator import Navigator
from serial_bridge import SerialBridge
from simulator import ArenaSimulator

def draw_hud(warped_display: np.ndarray, telemetry: dict, steering_data: dict, serial_status: str, fps: float):
    """Draws tactical telemetry HUD and vector overlays on the top-down arena view."""
    h, w, _ = warped_display.shape
    
    # Extract data
    my_bot = telemetry["my_bot"]
    enemy_bot = telemetry["enemy_bot"]
    cmd = steering_data["command"]
    dist = steering_data["distance"]
    err = steering_data["heading_error"]
    
    # 1. Draw My Robot
    if my_bot["detected"]:
        mx, my = int(my_bot["pos"][0]), int(my_bot["pos"][1])
        heading = my_bot["heading"]
        
        # Bot body circle (Green)
        cv2.circle(warped_display, (mx, my), 28, (0, 255, 0), 2)
        cv2.circle(warped_display, (mx, my), 4, (0, 255, 255), -1)
        
        # Heading arrow (45px long pointing in forward direction)
        rad = np.radians(heading)
        arrow_len = 45
        ax = int(mx + arrow_len * np.cos(rad))
        ay = int(my + arrow_len * np.sin(rad))
        cv2.arrowedLine(warped_display, (mx, my), (ax, ay), (0, 255, 255), 3, tipLength=0.3)
        cv2.putText(warped_display, f"MY BOT [{my_bot['source']}]", (mx - 55, my - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # 2. Draw Enemy Robot
    if enemy_bot["detected"]:
        ex, ey = int(enemy_bot["pos"][0]), int(enemy_bot["pos"][1])
        
        # Target crosshair (Red)
        cv2.circle(warped_display, (ex, ey), 30, (0, 0, 255), 2)
        cv2.circle(warped_display, (ex, ey), 5, (0, 0, 255), -1)
        cv2.line(warped_display, (ex - 35, ey), (ex + 35, ey), (0, 0, 255), 1)
        cv2.line(warped_display, (ex, ey - 35), (ex, ey + 35), (0, 0, 255), 1)
        cv2.putText(warped_display, f"ENEMY [{enemy_bot['source']}]", (ex - 55, ey - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # 3. Draw Target Vector Line (from My Bot to Enemy Bot)
    if my_bot["detected"] and enemy_bot["detected"]:
        mx, my = int(my_bot["pos"][0]), int(my_bot["pos"][1])
        ex, ey = int(enemy_bot["pos"][0]), int(enemy_bot["pos"][1])
        
        # Color line based on alignment: Green if aligned, Yellow if turning, Red if opposite
        line_color = (0, 255, 0) if abs(err) <= 20 else ((0, 255, 255) if abs(err) <= 60 else (0, 165, 255))
        cv2.line(warped_display, (mx, my), (ex, ey), line_color, 2, cv2.LINE_AA)
        
        # Display distance at midpoint
        mid_x, mid_y = (mx + ex) // 2, (my + ey) // 2
        cv2.putText(warped_display, f"{int(dist)}px", (mid_x + 8, mid_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # 4. Top Telemetry Banner
    banner_height = 80
    overlay = warped_display.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, warped_display, 0.15, 0, warped_display)
    
    # Steering command indicator color
    cmd_colors = {
        "FORWARD": (0, 255, 0),
        "RAM": (0, 0, 255),
        "TURN_LEFT": (255, 200, 0),
        "TURN_RIGHT": (255, 200, 0),
        "SEARCH": (120, 120, 120),
        "STOP": (0, 0, 255)
    }
    status_col = cmd_colors.get(cmd, (255, 255, 255))
    
    cv2.putText(warped_display, f"ACTION: {cmd}", (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.85, status_col, 2)
    cv2.putText(warped_display, f"ERR: {err:+.1f} deg | DIST: {int(dist)}px", (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    
    cv2.putText(warped_display, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    cv2.putText(warped_display, f"SERIAL: {serial_status}", (w - 240, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def build_composite_view(warped_view: np.ndarray, raw1: np.ndarray, raw2: np.ndarray,
                         pts1: list, pts2: list) -> np.ndarray:
    """Builds a side-by-side composite monitor with warped arena on the left and camera feeds on the right."""
    target_h = warped_view.shape[0]
    
    # Resize raw frames to fit stacked on right side
    side_w = int(target_h * 0.6)
    side_h = target_h // 2
    
    def prep_raw(frame, pts, label):
        if frame is None:
            f = np.zeros((side_h, side_w, 3), dtype=np.uint8)
            cv2.putText(f, f"{label} OFFLINE", (30, side_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return f
        f = cv2.resize(frame, (side_w, side_h))
        # Draw calibration polygon if points exist
        if pts and len(pts) == 4:
            scale_x = side_w / frame.shape[1]
            scale_y = side_h / frame.shape[0]
            scaled_pts = (np.array(pts) * [scale_x, scale_y]).astype(np.int32)
            cv2.polylines(f, [scaled_pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(f, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return f

    f1 = prep_raw(raw1, pts1, "CAM 1: LAPTOP WEBCAM (LIVE)" if raw1 is not None else "CAM 1 OFFLINE")
    f2 = prep_raw(raw2, pts2, "CAM 2: USB WEBCAM (LIVE)" if raw2 is not None else "CAM 2: USB (STANDBY)")
    
    right_stack = np.vstack([f1, f2])
    composite = np.hstack([warped_view, right_stack])
    return composite

def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Camera Combat Robot Tracker & ESP-NOW Steering")
    parser.add_argument("--simulate", action="store_true", help="Run with virtual arena simulation instead of physical webcams")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    parser.add_argument("--frames", type=int, default=0, help="Stop automatically after N frames (0 for infinite loop)")
    parser.add_argument("--record", type=str, default=None, help="Save output video to file (e.g. output.mp4)")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 65)
    print("      DUAL-CAMERA COMBAT ROBOT TRACKER & ESP-NOW STEERING      ")
    print("=" * 65)
    
    # Initialize Configs
    arena_cfg = ArenaConfig()
    cam_cfg = CameraConfig()
    track_cfg = TrackingConfig()
    steer_cfg = SteeringConfig()
    serial_cfg = SerialConfig()
    
    # Initialize Calibrator
    calibrator = ArenaCalibrator(arena_cfg.WARP_WIDTH, arena_cfg.WARP_HEIGHT)
    
    sim = None
    cam1 = None
    cam2 = None

    if args.simulate:
        print("\n[MODE] Running in VIRTUAL ARENA SIMULATION mode.")
        sim = ArenaSimulator(arena_cfg.WARP_WIDTH)
        # Apply virtual calibration directly
        calibrator.pts_cam1 = sim.cam1_corners
        calibrator.pts_cam2 = sim.cam2_corners
        calibrator.H_cam1, _ = cv2.findHomography(np.array(sim.cam1_corners, dtype=np.float32), calibrator.dst_pts)
        calibrator.H_cam2, _ = cv2.findHomography(np.array(sim.cam2_corners, dtype=np.float32), calibrator.dst_pts)
    else:
        # Physical camera capture with automatic laptop webcam detection
        print("\n[INFO] Detecting available camera devices...")
        available_cams = []
        for idx in range(3):
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                available_cams.append(idx)
                test_cap.release()

        print(f"[INFO] Discovered camera indices: {available_cams}")

        if len(available_cams) >= 2:
            print(f"[OK] Dual webcams found! Using index {available_cams[0]} (Cam 1) and index {available_cams[1]} (Cam 2)")
            cam1 = CameraStream(available_cams[0], cam_cfg.FRAME_WIDTH, cam_cfg.FRAME_HEIGHT, cam_cfg.FPS, "Cam 1")
            cam2 = CameraStream(available_cams[1], cam_cfg.FRAME_WIDTH, cam_cfg.FRAME_HEIGHT, cam_cfg.FPS, "Cam 2")
        elif len(available_cams) == 1:
            print(f"[OK] 1 Camera Found! Using Laptop Webcam (Index {available_cams[0]}) as Camera 1.")
            print("     Operating in SINGLE-CAMERA MODE (Cam 2 set to standby).")
            cam1 = CameraStream(available_cams[0], cam_cfg.FRAME_WIDTH, cam_cfg.FRAME_HEIGHT, cam_cfg.FPS, "Laptop Webcam")
            cam2 = None
        else:
            print("[WARN] No physical cameras detected. Falling back to VIRTUAL ARENA SIMULATOR.")
            sim = ArenaSimulator(arena_cfg.WARP_WIDTH)
            calibrator.pts_cam1 = sim.cam1_corners
            calibrator.pts_cam2 = sim.cam2_corners
            calibrator.H_cam1, _ = cv2.findHomography(np.array(sim.cam1_corners, dtype=np.float32), calibrator.dst_pts)
            calibrator.H_cam2, _ = cv2.findHomography(np.array(sim.cam2_corners, dtype=np.float32), calibrator.dst_pts)

        time.sleep(0.6)

    # Initialize Tracker, Navigator, Serial Bridge
    tracker = BotTracker(track_cfg, arena_cfg)
    navigator = Navigator(steer_cfg)
    bridge = SerialBridge(serial_cfg)
    
    fps_time = time.time()
    frame_counter = 0
    total_processed = 0
    current_fps = 0.0

    video_writer = None
    
    print("\n[SYSTEM RUNNING] Press 'q' or 'ESC' to exit.")
    if not sim:
        print("                 Keys: '1' Calib Cam 1 | '2' Calib Cam 2 | 'c' Full Calib")
    
    try:
        while True:
            # 1. Grab latest frames from either Simulator or physical Camera Streams
            if sim is not None:
                frame1, frame2 = sim.get_camera_frames()
                ret1, ret2 = True, True
                time.sleep(0.02)  # Maintain ~30-40 FPS simulated clock
            else:
                ret1, frame1 = cam1.read() if cam1 is not None else (False, None)
                ret2, frame2 = cam2.read() if cam2 is not None else (False, None)
            
            # Auto-initialize default calibration for live laptop camera if uncalibrated
            if calibrator.H_cam1 is None and ret1 and frame1 is not None:
                calibrator.set_default_camera_corners(1, frame1.shape[1], frame1.shape[0])
                
            # FPS Calculation
            frame_counter += 1
            total_processed += 1
            if time.time() - fps_time >= 1.0:
                current_fps = frame_counter / (time.time() - fps_time)
                frame_counter = 0
                fps_time = time.time()
                
            # 2. Check and Warp perspectives to top-down 2D floor view
            warped1 = calibrator.warp(frame1, cam_num=1) if ret1 else None
            warped2 = calibrator.warp(frame2, cam_num=2) if ret2 else None
            
            # Base warped canvas for visualization
            if warped1 is not None and warped2 is not None:
                arena_canvas = warped1.copy()
            elif warped1 is not None:
                arena_canvas = warped1.copy()
            elif warped2 is not None:
                arena_canvas = warped2.copy()
            else:
                arena_canvas = np.zeros((arena_cfg.WARP_HEIGHT, arena_cfg.WARP_WIDTH, 3), dtype=np.uint8)
                cv2.putText(arena_canvas, "CALIBRATION REQUIRED", (180, arena_cfg.WARP_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # 3. Track Bots & Fuse Coordinates (Camera Switching / Fallback)
            telemetry = tracker.process_and_fuse(warped1, warped2)
            
            # 4. Vector & Steering Math
            my_bot_info = telemetry["my_bot"]
            enemy_bot_info = telemetry["enemy_bot"]
            
            steering = navigator.compute_steering(
                my_pos=my_bot_info["pos"],
                my_heading_deg=my_bot_info["heading"],
                enemy_pos=enemy_bot_info["pos"]
            )
            
            # 5. Transmit packet over Serial to ESP32 ESP-NOW Master
            if serial_cfg.ENABLED and bridge.is_connected:
                bridge.send_packet(steering["packet"])
                serial_status = "ONLINE"
            elif serial_cfg.ENABLED:
                serial_status = "RECONNECTING"
            else:
                serial_status = "MOCK / READY"

            # Print telemetry summary every 30 frames
            if total_processed % 30 == 0:
                my_pos_str = f"({int(my_bot_info['pos'][0])}, {int(my_bot_info['pos'][1])})" if my_bot_info['detected'] else "LOST"
                enemy_pos_str = f"({int(enemy_bot_info['pos'][0])}, {int(enemy_bot_info['pos'][1])})" if enemy_bot_info['detected'] else "LOST"
                print(f"[TRACK] Frame {total_processed} | MyBot: {my_pos_str} (Hdg: {my_bot_info['heading'] or 0:.0f}°) | Enemy: {enemy_pos_str} [{enemy_bot_info['source']}] | Cmd: {steering['command']} | Packet: {steering['packet'].strip()}")
                
            # 6. Render Tactical HUD & Composite Multi-Camera View
            draw_hud(arena_canvas, telemetry, steering, serial_status, current_fps)
            composite_view = build_composite_view(
                arena_canvas, frame1, frame2,
                calibrator.pts_cam1, calibrator.pts_cam2
            )
            
            # Record video if requested
            if args.record:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    ch, cw, _ = composite_view.shape
                    video_writer = cv2.VideoWriter(args.record, fourcc, 30.0, (cw, ch))
                video_writer.write(composite_view)

            if not args.headless:
                cv2.imshow("Combat Arena Tactical Ground-Truth Monitor", composite_view)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
                elif key == ord('1') and frame1 is not None:
                    print("\n[CALIB] Starting Camera 1 calibration...")
                    calibrator.calibrate_camera(1, frame1)
                elif key == ord('2') and frame2 is not None:
                    print("\n[CALIB] Starting Camera 2 calibration...")
                    calibrator.calibrate_camera(2, frame2)
                elif key == ord('c'):
                    if frame1 is not None:
                        calibrator.calibrate_camera(1, frame1)
                    if frame2 is not None:
                        calibrator.calibrate_camera(2, frame2)
            else:
                # Brief sleep in headless mode
                time.sleep(0.01)

            if args.frames > 0 and total_processed >= args.frames:
                print(f"\n[INFO] Reached requested limit of {args.frames} frames.")
                # Save a high-resolution snapshot
                cv2.imwrite("tactical_dashboard_snapshot.png", composite_view)
                print("[OK] Saved live snapshot -> tactical_dashboard_snapshot.png")
                break
                    
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Shutting down...")
    finally:
        print("\n[INFO] Cleaning up resources...")
        if video_writer is not None:
            video_writer.release()
            print(f"[OK] Saved recorded video to {args.record}")
        bridge.close()
        if cam1 is not None:
            cam1.stop()
        if cam2 is not None:
            cam2.stop()
        cv2.destroyAllWindows()
        print("[OK] System shutdown complete.")

if __name__ == "__main__":
    main()

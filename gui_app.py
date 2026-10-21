"""
Combat Robot Arena Tracker - GUI Application (Tkinter + OpenCV + PIL).
Provides a dedicated Windows desktop application with live camera feeds,
warped top-down view, visual vector steering, live telemetry, and interactive controls.
"""

import sys
import os
import time
import math
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Import tracker modules
from config import ArenaConfig, CameraConfig, TrackingConfig, SteeringConfig, SerialConfig
from camera_stream import CameraStream
from homography import ArenaCalibrator
from tracker import BotTracker
from navigator import Navigator
from serial_bridge import SerialBridge
from simulator import ArenaSimulator

class CombatTrackerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Combat Robot Arena Autonomous Tracker & ESP-NOW Link")
        self.root.geometry("1280x820")
        self.root.configure(bg="#121216")
        
        # Bring to front on launch
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        # Configs & Subsystems
        self.arena_cfg = ArenaConfig()
        self.cam_cfg = CameraConfig()
        self.track_cfg = TrackingConfig()
        self.steer_cfg = SteeringConfig()
        self.serial_cfg = SerialConfig()
        
        self.calibrator = ArenaCalibrator(self.arena_cfg.WARP_WIDTH, self.arena_cfg.WARP_HEIGHT)
        self.tracker = BotTracker(self.track_cfg, self.arena_cfg)
        self.navigator = Navigator(self.steer_cfg)
        self.serial_bridge = SerialBridge(self.serial_cfg)
        self.simulator = ArenaSimulator(self.arena_cfg.WARP_WIDTH)
        
        # Mode: "WEBCAM" or "SIMULATION"
        self.mode_var = tk.StringVar(value="WEBCAM")
        self.cam1 = None
        self.cam2 = None
        self.is_running = True
        
        # Interactive Calibration State
        self.calibrating_cam = 0  # 0: None, 1: Cam 1, 2: Cam 2
        self.calib_clicks = []
        
        # Build UI layout
        self.setup_ui()
        
        # Try initializing webcams; fallback to simulation if none found
        self.init_cameras()
        
        # Start frame update loop
        self.fps_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.update_loop()

    def setup_ui(self):
        """Builds modern dark-mode GUI layout."""
        # Top Header Bar
        header = tk.Frame(self.root, bg="#1a1a24", height=50)
        header.pack(fill=tk.X, side=tk.TOP)
        
        lbl_title = tk.Label(header, text="COMBAT ROBOT AUTONOMOUS TRACKER", 
                             font=("Segoe UI", 14, "bold"), fg="#00ffcc", bg="#1a1a24")
        lbl_title.pack(side=tk.LEFT, padx=16, pady=10)
        
        # Mode Switch Radio Buttons
        mode_frame = tk.Frame(header, bg="#1a1a24")
        mode_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(mode_frame, text="Source:", font=("Segoe UI", 10), fg="#aaaaaa", bg="#1a1a24").pack(side=tk.LEFT, padx=4)
        r1 = tk.Radiobutton(mode_frame, text="Laptop/USB Webcam", variable=self.mode_var, value="WEBCAM",
                            command=self.switch_mode, font=("Segoe UI", 9, "bold"),
                            fg="#00ffcc", selectcolor="#252535", bg="#1a1a24", activebackground="#1a1a24")
        r1.pack(side=tk.LEFT, padx=6)
        
        r2 = tk.Radiobutton(mode_frame, text="Virtual Arena Simulation", variable=self.mode_var, value="SIMULATION",
                            command=self.switch_mode, font=("Segoe UI", 9, "bold"),
                            fg="#ffaa00", selectcolor="#252535", bg="#1a1a24", activebackground="#1a1a24")
        r2.pack(side=tk.LEFT, padx=6)

        # Status Badges
        self.lbl_fps = tk.Label(header, text="FPS: 0.0", font=("Consolas", 10, "bold"), fg="#00ff88", bg="#1a1a24")
        self.lbl_fps.pack(side=tk.RIGHT, padx=16)

        self.lbl_cam_status = tk.Label(header, text="Webcam: Detecting...", font=("Segoe UI", 9), fg="#ffffff", bg="#1a1a24")
        self.lbl_cam_status.pack(side=tk.RIGHT, padx=12)

        # Main Body Splitter: Left (Video Canvas) + Right (Control & Telemetry Panel)
        body = tk.Frame(self.root, bg="#121216")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Video Container
        video_frame = tk.Frame(body, bg="#1a1a24", bd=1, relief=tk.SOLID)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Canvas for displaying composite OpenCV stream
        self.canvas = tk.Canvas(video_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Right Telemetry and Controls Panel
        panel = tk.Frame(body, bg="#181820", width=340)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        # Command Action Card
        cmd_card = tk.LabelFrame(panel, text="AUTONOMOUS STEERING ACTION", font=("Segoe UI", 9, "bold"),
                                 fg="#aaaaaa", bg="#20202c", bd=1, padx=10, pady=8)
        cmd_card.pack(fill=tk.X, padx=10, pady=(10, 8))

        self.lbl_action = tk.Label(cmd_card, text="SEARCH", font=("Segoe UI", 20, "bold"),
                                   fg="#ffbb00", bg="#20202c")
        self.lbl_action.pack(pady=4)

        # Telemetry Metrics Table
        tele_card = tk.LabelFrame(panel, text="LIVE ROBOT TELEMETRY", font=("Segoe UI", 9, "bold"),
                                  fg="#aaaaaa", bg="#20202c", bd=1, padx=10, pady=8)
        tele_card.pack(fill=tk.X, padx=10, pady=6)

        def make_row(parent, title):
            row = tk.Frame(parent, bg="#20202c")
            row.pack(fill=tk.X, pady=2)
            lbl = tk.Label(row, text=title, font=("Segoe UI", 9), fg="#9999aa", bg="#20202c")
            lbl.pack(side=tk.LEFT)
            val = tk.Label(row, text="--", font=("Consolas", 10, "bold"), fg="#ffffff", bg="#20202c")
            val.pack(side=tk.RIGHT)
            return val

        self.val_my_pos = make_row(tele_card, "My Bot (X, Y):")
        self.val_my_hdg = make_row(tele_card, "My Heading:")
        self.val_enemy_pos = make_row(tele_card, "Enemy (X, Y):")
        self.val_distance = make_row(tele_card, "Target Distance:")
        self.val_err = make_row(tele_card, "Heading Error:")
        self.val_motors = make_row(tele_card, "Left / Right PWM:")
        self.val_source = make_row(tele_card, "Active Tracking Cam:")

        # Serial / ESP-NOW Output Monitor
        serial_card = tk.LabelFrame(panel, text="ESP-NOW SERIAL PACKET (MASTER)", font=("Segoe UI", 9, "bold"),
                                    fg="#aaaaaa", bg="#20202c", bd=1, padx=10, pady=8)
        serial_card.pack(fill=tk.X, padx=10, pady=6)

        self.lbl_serial_port = tk.Label(serial_card, text=f"Port: {self.serial_cfg.PORT} @ {self.serial_cfg.BAUD_RATE}",
                                        font=("Segoe UI", 8), fg="#00ffcc", bg="#20202c")
        self.lbl_serial_port.pack(anchor=tk.W)

        self.lbl_packet = tk.Label(serial_card, text="<SEARCH,0,0,0,0>", font=("Consolas", 9, "bold"),
                                   fg="#00ff88", bg="#121218", padx=6, pady=4, relief=tk.SUNKEN)
        self.lbl_packet.pack(fill=tk.X, pady=4)

        # Controls & Calibration Buttons
        btn_card = tk.LabelFrame(panel, text="CAMERA CALIBRATION & UTILITIES", font=("Segoe UI", 9, "bold"),
                                 fg="#aaaaaa", bg="#20202c", bd=1, padx=10, pady=8)
        btn_card.pack(fill=tk.X, padx=10, pady=6)

        btn_calib1 = tk.Button(btn_card, text="🎯 Calibrate Cam 1 Corners", font=("Segoe UI", 9, "bold"),
                               bg="#2d2d40", fg="#ffffff", activebackground="#3d3d55",
                               command=lambda: self.start_calibration(1))
        btn_calib1.pack(fill=tk.X, pady=3)

        btn_calib2 = tk.Button(btn_card, text="🎯 Calibrate Cam 2 Corners", font=("Segoe UI", 9),
                               bg="#2d2d40", fg="#ffffff", activebackground="#3d3d55",
                               command=lambda: self.start_calibration(2))
        btn_calib2.pack(fill=tk.X, pady=3)

        btn_reset_calib = tk.Button(btn_card, text="🔄 Reset to Default Arena", font=("Segoe UI", 8),
                                    bg="#222230", fg="#aaaaaa", command=self.reset_calibration)
        btn_reset_calib.pack(fill=tk.X, pady=3)

        btn_aruco = tk.Button(btn_card, text="📄 Show My ArUco Tag", font=("Segoe UI", 9),
                              bg="#203525", fg="#00ff88", command=self.show_aruco_tag_popup)
        btn_aruco.pack(fill=tk.X, pady=3)

        # E-STOP Emergency Stop Button
        btn_estop = tk.Button(panel, text="🛑 EMERGENCY STOP (E-STOP)", font=("Segoe UI", 11, "bold"),
                              bg="#aa1122", fg="#ffffff", activebackground="#dd2233",
                              command=self.emergency_stop)
        btn_estop.pack(fill=tk.X, padx=10, pady=(10, 6))

    def init_cameras(self):
        """Probes and initializes cameras."""
        print("[INFO] Probing camera devices with DirectShow...")
        available = []
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
                
        if len(available) >= 2:
            self.cam1 = CameraStream(available[0], self.cam_cfg.FRAME_WIDTH, self.cam_cfg.FRAME_HEIGHT, self.cam_cfg.FPS, "Cam 1")
            self.cam2 = CameraStream(available[1], self.cam_cfg.FRAME_WIDTH, self.cam_cfg.FRAME_HEIGHT, self.cam_cfg.FPS, "Cam 2")
            self.lbl_cam_status.config(text=f"Dual Webcams: Cam {available[0]} & {available[1]}", fg="#00ff88")
        elif len(available) == 1:
            self.cam1 = CameraStream(available[0], self.cam_cfg.FRAME_WIDTH, self.cam_cfg.FRAME_HEIGHT, self.cam_cfg.FPS, "Laptop Webcam")
            self.cam2 = None
            self.lbl_cam_status.config(text=f"Laptop Webcam (Index {available[0]}) Active", fg="#00ffcc")
        else:
            self.cam1 = None
            self.cam2 = None
            self.mode_var.set("SIMULATION")
            self.lbl_cam_status.config(text="Virtual Arena Simulation Mode", fg="#ffaa00")

    def switch_mode(self):
        mode = self.mode_var.get()
        if mode == "WEBCAM":
            if self.cam1 is None:
                self.init_cameras()
        else:
            self.lbl_cam_status.config(text="Virtual Simulation Active", fg="#ffaa00")

    def start_calibration(self, cam_num: int):
        self.calibrating_cam = cam_num
        self.calib_clicks = []
        messagebox.showinfo("Interactive Calibration", 
                            f"Click the 4 corners of the arena floor on the Camera {cam_num} feed:\n"
                            "1: Top-Left -> 2: Top-Right -> 3: Bottom-Right -> 4: Bottom-Left.")

    def reset_calibration(self):
        self.calibrator.H_cam1 = None
        self.calibrator.H_cam2 = None
        messagebox.showinfo("Calibration Reset", "Arena perspective reset to default bounds.")

    def on_canvas_click(self, event):
        """Handles mouse clicks on the canvas for interactive corner selection."""
        if self.calibrating_cam == 0:
            return
            
        x, y = event.x, event.y
        self.calib_clicks.append([x, y])
        
        if len(self.calib_clicks) == 4:
            src = np.array(self.calib_clicks, dtype=np.float32)
            H, _ = cv2.findHomography(src, self.calibrator.dst_pts)
            if self.calibrating_cam == 1:
                self.calibrator.pts_cam1 = self.calib_clicks
                self.calibrator.H_cam1 = H
            else:
                self.calibrator.pts_cam2 = self.calib_clicks
                self.calibrator.H_cam2 = H
            self.calibrating_cam = 0
            self.calibrator.save_calibration()
            messagebox.showinfo("Calibration Saved", "Corner calibration saved successfully!")

    def show_aruco_tag_popup(self):
        """Displays the ArUco marker on screen for immediate testing."""
        popup = tk.Toplevel(self.root)
        popup.title("My Robot ArUco Tag (DICT_4X4_50, ID 0)")
        popup.geometry("380x420")
        popup.configure(bg="#ffffff")
        
        tag_path = os.path.join(os.path.dirname(__file__), "my_bot_aruco_tag.png")
        if os.path.exists(tag_path):
            img = Image.open(tag_path).resize((320, 320))
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(popup, image=photo, bg="#ffffff")
            lbl.image = photo
            lbl.pack(pady=10)
            tk.Label(popup, text="Hold this marker in front of your webcam to test tracking!",
                     font=("Segoe UI", 9, "bold"), fg="#111111", bg="#ffffff").pack()

    def emergency_stop(self):
        """Sends immediate stop packet over serial and flags E-STOP."""
        self.lbl_action.config(text="E-STOP ACTIVE", fg="#ff0000")
        if self.serial_bridge.is_connected:
            self.serial_bridge.send_packet("<STOP,0,0,0,0>\n")
        self.lbl_packet.config(text="<STOP,0,0,0,0>", fg="#ff2222")

    def update_loop(self):
        """Main real-time UI animation loop."""
        if not self.is_running:
            return

        mode = self.mode_var.get()
        frame1, frame2 = None, None
        
        if mode == "SIMULATION":
            frame1, frame2 = self.simulator.get_camera_frames()
            ret1, ret2 = True, True
        else:
            ret1, frame1 = self.cam1.read() if self.cam1 is not None else (False, None)
            ret2, frame2 = self.cam2.read() if self.cam2 is not None else (False, None)

        # Auto-initialize default calibration if uncalibrated
        if self.calibrator.H_cam1 is None and ret1 and frame1 is not None:
            self.calibrator.set_default_camera_corners(1, frame1.shape[1], frame1.shape[0])

        # Warp perspectives to top-down 2D floor view
        warped1 = self.calibrator.warp(frame1, cam_num=1) if ret1 else None
        warped2 = self.calibrator.warp(frame2, cam_num=2) if ret2 else None

        if warped1 is not None:
            arena_canvas = warped1.copy()
        elif warped2 is not None:
            arena_canvas = warped2.copy()
        else:
            arena_canvas = np.zeros((self.arena_cfg.WARP_HEIGHT, self.arena_cfg.WARP_WIDTH, 3), dtype=np.uint8)

        # Track Bots & Fuse Coordinates
        telemetry = self.tracker.process_and_fuse(warped1, warped2)
        my_bot = telemetry["my_bot"]
        enemy_bot = telemetry["enemy_bot"]

        # Vector Math & Steering
        steering = self.navigator.compute_steering(
            my_pos=my_bot["pos"],
            my_heading_deg=my_bot["heading"],
            enemy_pos=enemy_bot["pos"]
        )

        # Serial Transmission
        if self.serial_cfg.ENABLED and self.serial_bridge.is_connected:
            self.serial_bridge.send_packet(steering["packet"])
            serial_stat = "ONLINE"
        else:
            serial_stat = "MOCK / READY"

        # Update FPS
        self.frame_count += 1
        now = time.time()
        if now - self.fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.fps_time)
            self.frame_count = 0
            self.fps_time = now
            self.lbl_fps.config(text=f"FPS: {self.fps:.1f}")

        # Draw Visual Overlays (HUD, Vectors, Bot Rings)
        self.draw_overlays(arena_canvas, telemetry, steering)

        # Compose Side-by-Side Video Monitor
        composite = self.build_composite_view(arena_canvas, frame1, frame2)

        # Render onto Tkinter Canvas
        self.display_image_on_canvas(composite)

        # Update Telemetry Sidebar Values
        self.update_telemetry_sidebar(my_bot, enemy_bot, steering)

        # Schedule next update (aim for ~35-40 FPS)
        self.root.after(25, self.update_loop)

    def draw_overlays(self, canvas: np.ndarray, telemetry: dict, steering: dict):
        """Draws heading vectors, target crosshairs, and distance line."""
        my_bot = telemetry["my_bot"]
        enemy_bot = telemetry["enemy_bot"]
        err = steering["heading_error"]
        dist = steering["distance"]

        # My Bot
        if my_bot["detected"]:
            mx, my = int(my_bot["pos"][0]), int(my_bot["pos"][1])
            hdg = my_bot["heading"]
            cv2.circle(canvas, (mx, my), 28, (0, 255, 0), 2)
            cv2.circle(canvas, (mx, my), 4, (0, 255, 255), -1)
            
            # Heading arrow
            rad = np.radians(hdg)
            ax = int(mx + 45 * np.cos(rad))
            ay = int(my + 45 * np.sin(rad))
            cv2.arrowedLine(canvas, (mx, my), (ax, ay), (0, 255, 255), 3, tipLength=0.3)
            cv2.putText(canvas, "MY BOT", (mx - 30, my - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Enemy Bot
        if enemy_bot["detected"]:
            ex, ey = int(enemy_bot["pos"][0]), int(enemy_bot["pos"][1])
            cv2.circle(canvas, (ex, ey), 30, (0, 0, 255), 2)
            cv2.circle(canvas, (ex, ey), 5, (0, 0, 255), -1)
            cv2.line(canvas, (ex - 35, ey), (ex + 35, ey), (0, 0, 255), 1)
            cv2.line(canvas, (ex, ey - 35), (ex, ey + 35), (0, 0, 255), 1)
            cv2.putText(canvas, "ENEMY", (ex - 28, ey - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Target Vector
        if my_bot["detected"] and enemy_bot["detected"]:
            mx, my = int(my_bot["pos"][0]), int(my_bot["pos"][1])
            ex, ey = int(enemy_bot["pos"][0]), int(enemy_bot["pos"][1])
            color = (0, 255, 0) if abs(err) <= 20 else ((0, 255, 255) if abs(err) <= 60 else (0, 165, 255))
            cv2.line(canvas, (mx, my), (ex, ey), color, 2, cv2.LINE_AA)
            mid_x, mid_y = (mx + ex) // 2, (my + ey) // 2
            cv2.putText(canvas, f"{int(dist)}px", (mid_x + 8, mid_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def build_composite_view(self, warped_view: np.ndarray, raw1: np.ndarray, raw2: np.ndarray) -> np.ndarray:
        target_h = warped_view.shape[0]
        side_w = int(target_h * 0.55)
        side_h = target_h // 2
        
        def prep(frame, label):
            if frame is None:
                f = np.zeros((side_h, side_w, 3), dtype=np.uint8)
                cv2.putText(f, label, (20, side_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
                return f
            f = cv2.resize(frame, (side_w, side_h))
            cv2.putText(f, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            return f

        f1 = prep(raw1, "CAM 1: WEBCAM (LIVE)" if raw1 is not None else "CAM 1 OFFLINE")
        f2 = prep(raw2, "CAM 2: USB WEBCAM (LIVE)" if raw2 is not None else "CAM 2 (STANDBY)")
        right = np.vstack([f1, f2])
        return np.hstack([warped_view, right])

    def display_image_on_canvas(self, bgr_img: np.ndarray):
        """Converts OpenCV BGR image and renders it fitted to Tkinter canvas."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 50 or canvas_h < 50:
            return

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        
        # Scale proportionally to fit canvas
        img_w, img_h = im.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w, new_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
        
        im_resized = im.resize((new_w, new_h), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(image=im_resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.photo, anchor=tk.CENTER)

    def update_telemetry_sidebar(self, my_bot: dict, enemy_bot: dict, steering: dict):
        cmd = steering["command"]
        cmd_colors = {
            "FORWARD": "#00ff88",
            "RAM": "#ff3333",
            "TURN_LEFT": "#ffcc00",
            "TURN_RIGHT": "#ffcc00",
            "SEARCH": "#8888aa",
            "STOP": "#ff2222"
        }
        self.lbl_action.config(text=cmd, fg=cmd_colors.get(cmd, "#ffffff"))

        my_str = f"({int(my_bot['pos'][0])}, {int(my_bot['pos'][1])})" if my_bot["detected"] else "LOST"
        hdg_str = f"{my_bot['heading']:.1f}°" if my_bot["heading"] is not None else "--"
        enemy_str = f"({int(enemy_bot['pos'][0])}, {int(enemy_bot['pos'][1])})" if enemy_bot["detected"] else "LOST"

        self.val_my_pos.config(text=my_str)
        self.val_my_hdg.config(text=hdg_str)
        self.val_enemy_pos.config(text=enemy_str)
        self.val_distance.config(text=f"{steering['distance']} px")
        self.val_err.config(text=f"{steering['heading_error']:+.1f}°")
        self.val_motors.config(text=f"{steering['throttle_left']} / {steering['throttle_right']}")
        self.val_source.config(text=enemy_bot["source"])

        self.lbl_packet.config(text=steering["packet"].strip())

    def on_close(self):
        self.is_running = False
        if self.cam1 is not None:
            self.cam1.stop()
        if self.cam2 is not None:
            self.cam2.stop()
        self.serial_bridge.close()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = CombatTrackerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()

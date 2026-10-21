"""
MASTER LAPTOP TACTICAL COMMAND DASHBOARD (ASUS TUF 1 - Corner A + ESP-NOW Link).
Professional Combat Robotics Ground-Truth Visualizer & AI Co-Pilot.

Key Features:
- Crash-proof CameraStream with watchdog and auto-reconnect
- Autonomous Weak-Spot Discovery (AI scans camera feed, finds rear/wheels & targets automatically)
- AI Steer-by-Wire (User drives Front/Back with W/S; AI aims Left/Right into weak spots!)
- Dedicated "AI COMBAT CO-PILOT" Tab with real-time driving tips, coaching & diagnostic error monitor
- Dual-Camera Live Feeds: Displays BOTH Corner A (Local) and Corner B (Streamed from Laptop 2)
- Photo Upload for Our Bot only (all other blobs = enemy)
- Master E-STOP Kill Switch & PySerial ESP-NOW Transmission
"""

import os
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

from config import ArenaConfig, CameraConfig, VisualTrackingConfig, NetworkConfig, SteeringConfig, SerialConfig, TacticsConfig
from camera_stream import CameraStream
from homography import ArenaCalibrator
from visual_tracker import VisualRobotTracker
from navigator import Navigator
from serial_bridge import SerialBridge
from network_bridge import UDPCornerSender, UDPCornerReceiver

class MasterCommandDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("APEX COMBAT ROBOTICS - AI COMMAND & CO-PILOT SYSTEM")
        self.root.geometry("1440x920")
        self.root.configure(bg="#0b0c10")

        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        # Configs
        self.arena_cfg = ArenaConfig()
        self.cam_cfg = CameraConfig()
        self.track_cfg = VisualTrackingConfig()
        self.steer_cfg = SteeringConfig()
        self.tactics_cfg = TacticsConfig()
        self.serial_cfg = SerialConfig()
        self.net_cfg = NetworkConfig()

        # Core Systems
        self.calibrator = ArenaCalibrator(self.arena_cfg.WARP_WIDTH, self.arena_cfg.WARP_HEIGHT, "calibration_master.json")
        self.tracker = VisualRobotTracker(self.track_cfg, self.arena_cfg)
        self.navigator = Navigator(self.steer_cfg, self.tactics_cfg)
        self.serial_bridge = SerialBridge(self.serial_cfg)
        
        # Dual-Video UDP Network Bridge
        self.udp_receiver = UDPCornerReceiver(listen_port=self.net_cfg.UDP_PORT)
        self.udp_sender = None

        self.cam = None
        self.is_running = True
        self.is_paused = False
        self.estop_active = False

        # === CO-PILOT STEER-BY-WIRE STATE ===
        # User controls front/back throttle; AI controls left/right steering!
        self.control_mode = tk.StringVar(value="STEER_ASSIST")  # "STEER_ASSIST" or "FULL_AUTO"
        self.user_throttle_power = 220
        self.user_throttle_active = 0
        self.key_fwd_held = False
        self.key_rev_held = False

        # AI Diagnostic Event Log buffer
        self.ai_log_entries = []

        self.setup_ui()
        self.init_camera()
        self.setup_keyboard_controls()

        self.fps_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.update_loop()

    def setup_keyboard_controls(self):
        """Binds W/S and Arrow keys for Steer-by-Wire throttle."""
        self.root.bind("<KeyPress-w>", self.on_fwd_press)
        self.root.bind("<KeyPress-W>", self.on_fwd_press)
        self.root.bind("<KeyPress-Up>", self.on_fwd_press)
        self.root.bind("<KeyRelease-w>", self.on_fwd_release)
        self.root.bind("<KeyRelease-W>", self.on_fwd_release)
        self.root.bind("<KeyRelease-Up>", self.on_fwd_release)

        self.root.bind("<KeyPress-s>", self.on_rev_press)
        self.root.bind("<KeyPress-S>", self.on_rev_press)
        self.root.bind("<KeyPress-Down>", self.on_rev_press)
        self.root.bind("<KeyRelease-s>", self.on_rev_release)
        self.root.bind("<KeyRelease-S>", self.on_rev_release)
        self.root.bind("<KeyRelease-Down>", self.on_rev_release)

        self.root.bind("<space>", lambda e: self.brake())

    def on_fwd_press(self, event=None):
        self.key_fwd_held = True
        self.user_throttle_active = self.user_throttle_power

    def on_fwd_release(self, event=None):
        self.key_fwd_held = False
        if not self.key_rev_held:
            self.user_throttle_active = 0

    def on_rev_press(self, event=None):
        self.key_rev_held = True
        self.user_throttle_active = -int(self.user_throttle_power * 0.85)

    def on_rev_release(self, event=None):
        self.key_rev_held = False
        if not self.key_fwd_held:
            self.user_throttle_active = 0

    def brake(self):
        self.user_throttle_active = 0
        self.key_fwd_held = False
        self.key_rev_held = False

    def setup_ui(self):
        # 1. Top Header & Tab Navigation Bar
        header = tk.Frame(self.root, bg="#13151f", height=56, bd=0)
        header.pack(fill=tk.X, side=tk.TOP)

        title_frame = tk.Frame(header, bg="#13151f")
        title_frame.pack(side=tk.LEFT, padx=16, pady=6)

        tk.Label(title_frame, text="APEX COMBAT GROUND TRUTH", font=("Impact", 15),
                 fg="#00e5ff", bg="#13151f").pack(side=tk.LEFT)
        tk.Label(title_frame, text="  [MASTER CONTROLLER - CORNER A]", font=("Segoe UI", 9, "bold"),
                 fg="#8892b0", bg="#13151f").pack(side=tk.LEFT)

        # Tab Navigation Pill Buttons
        tabs_frame = tk.Frame(header, bg="#13151f")
        tabs_frame.pack(side=tk.LEFT, padx=24)

        self.btn_tab_arena = tk.Button(tabs_frame, text="🎮 ARENA VISION & HUD", font=("Segoe UI", 9, "bold"),
                                       bg="#00e5ff", fg="#000000", relief=tk.FLAT, padx=12, pady=4,
                                       command=lambda: self.switch_tab("ARENA"))
        self.btn_tab_arena.pack(side=tk.LEFT, padx=4)

        self.btn_tab_ai = tk.Button(tabs_frame, text="🤖 AI COMBAT CO-PILOT & COACH", font=("Segoe UI", 9, "bold"),
                                    bg="#1c2536", fg="#00e5ff", relief=tk.FLAT, padx=12, pady=4,
                                    command=lambda: self.switch_tab("AI"))
        self.btn_tab_ai.pack(side=tk.LEFT, padx=4)

        # Status Badges on Right
        self.lbl_fps = tk.Label(header, text="30.0 FPS", font=("Consolas", 10, "bold"), fg="#00ff88", bg="#13151f")
        self.lbl_fps.pack(side=tk.RIGHT, padx=16)

        self.lbl_net_status = tk.Label(header, text="● LAPTOP 2: WAITING FOR LINK",
                                       font=("Segoe UI", 9, "bold"), fg="#ffaa00", bg="#13151f")
        self.lbl_net_status.pack(side=tk.RIGHT, padx=14)

        self.lbl_serial_badge = tk.Label(header, text="● ESP-NOW: READY", font=("Segoe UI", 9, "bold"),
                                         fg="#00e5ff", bg="#13151f")
        self.lbl_serial_badge.pack(side=tk.RIGHT, padx=14)

        # Main Container holding both Tabs
        self.container = tk.Frame(self.root, bg="#0b0c10")
        self.container.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # ==========================================
        # TAB 1: ARENA VISION & COMBAT HUD VIEW
        # ==========================================
        self.frame_tab_arena = tk.Frame(self.container, bg="#0b0c10")
        self.frame_tab_arena.pack(fill=tk.BOTH, expand=True)

        # Video Display Canvas (Left Side of Tab 1)
        video_frame = tk.Frame(self.frame_tab_arena, bg="#141622", bd=1, relief=tk.SOLID)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(video_frame, bg="#050608", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Control Panel (Right Side of Tab 1)
        panel = tk.Frame(self.frame_tab_arena, bg="#11131c", width=360)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        # Emergency Stop Button
        self.btn_estop = tk.Button(panel, text="🛑 EMERGENCY STOP (E-STOP)", font=("Segoe UI", 12, "bold"),
                                   bg="#e60026", fg="#ffffff", activebackground="#ff1a3c",
                                   relief=tk.FLAT, pady=8, cursor="hand2", command=self.toggle_estop)
        self.btn_estop.pack(fill=tk.X, padx=10, pady=(8, 4))

        # Autonomous Steering Action Card
        action_card = tk.LabelFrame(panel, text="ACTIVE STEERING STATUS", font=("Segoe UI", 8, "bold"),
                                    fg="#64ffda", bg="#161926", bd=1, padx=8, pady=2)
        action_card.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_action = tk.Label(action_card, text="SEARCHING", font=("Impact", 20),
                                   fg="#ffb703", bg="#161926")
        self.lbl_action.pack(pady=1)

        # Co-Pilot Drive Mode Card
        drive_card = tk.LabelFrame(panel, text="STEER-BY-WIRE DRIVE MODE", font=("Segoe UI", 8, "bold"),
                                   fg="#00e5ff", bg="#161926", bd=1, padx=8, pady=4)
        drive_card.pack(fill=tk.X, padx=10, pady=3)

        tk.Radiobutton(drive_card, text="🤖 AI Steer-by-Wire (You: W/S | AI: Left/Right Aim)",
                       variable=self.control_mode, value="STEER_ASSIST",
                       font=("Segoe UI", 8, "bold"), fg="#00ffcc", bg="#161926",
                       selectcolor="#090a0f", activebackground="#161926").pack(anchor=tk.W)

        tk.Radiobutton(drive_card, text="🚀 Full Autonomous (AI controls Throttle & Steering)",
                       variable=self.control_mode, value="FULL_AUTO",
                       font=("Segoe UI", 8), fg="#8892b0", bg="#161926",
                       selectcolor="#090a0f", activebackground="#161926").pack(anchor=tk.W)

        # Quick Throttle Push Buttons on Panel
        btn_drive_row = tk.Frame(drive_card, bg="#161926")
        btn_drive_row.pack(fill=tk.X, pady=3)

        self.btn_fwd = tk.Button(btn_drive_row, text="⬆️ FWD [W]", font=("Segoe UI", 8, "bold"),
                                 bg="#1f3b2b", fg="#00ff88", activebackground="#2a523c")
        self.btn_fwd.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.btn_fwd.bind("<ButtonPress-1>", lambda e: self.on_fwd_press())
        self.btn_fwd.bind("<ButtonRelease-1>", lambda e: self.on_fwd_release())

        self.btn_rev = tk.Button(btn_drive_row, text="⬇️ REV [S]", font=("Segoe UI", 8, "bold"),
                                 bg="#3b2b1f", fg="#ffaa00", activebackground="#523c2a")
        self.btn_rev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.btn_rev.bind("<ButtonPress-1>", lambda e: self.on_rev_press())
        self.btn_rev.bind("<ButtonRelease-1>", lambda e: self.on_rev_release())

        self.btn_stop = tk.Button(btn_drive_row, text="🛑 BRAKE", font=("Segoe UI", 8, "bold"),
                                  bg="#3b1f23", fg="#ff4455", command=self.brake)
        self.btn_stop.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # Robot Profile (OUR BOT ONLY)
        bot_card = tk.LabelFrame(panel, text="OUR COMBAT ROBOT PROFILE", font=("Segoe UI", 8, "bold"),
                                 fg="#00e5ff", bg="#161926", bd=1, padx=8, pady=3)
        bot_card.pack(fill=tk.X, padx=10, pady=2)

        thumb_row = tk.Frame(bot_card, bg="#161926")
        thumb_row.pack(fill=tk.X, pady=1)

        self.lbl_bot_thumb = tk.Label(thumb_row, bg="#090a0f", width=50, height=50, bd=1, relief=tk.SOLID)
        self.lbl_bot_thumb.pack(side=tk.LEFT, padx=4)

        desc_col = tk.Frame(thumb_row, bg="#161926")
        desc_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        tk.Label(desc_col, text="Template Match Active", font=("Segoe UI", 8, "bold"), fg="#00ff88", bg="#161926").pack(anchor=tk.W)
        tk.Label(desc_col, text="Enemy found automatically", font=("Segoe UI", 8), fg="#8892b0", bg="#161926").pack(anchor=tk.W)

        btn_row = tk.Frame(bot_card, bg="#161926")
        btn_row.pack(fill=tk.X, pady=2)

        tk.Button(btn_row, text="📁 Upload Photo...", font=("Segoe UI", 8, "bold"),
                  bg="#1f2937", fg="#00e5ff", command=self.upload_bot_photo).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        tk.Button(btn_row, text="✂️ Crop Live", font=("Segoe UI", 8, "bold"),
                  bg="#1f2937", fg="#ffffff", command=self.crop_bot_photo).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        # Arena Calibration Card (1-Click Presets)
        calib_card = tk.LabelFrame(panel, text="ARENA CALIBRATION PRESETS", font=("Segoe UI", 8, "bold"),
                                   fg="#ffaa00", bg="#161926", bd=1, padx=8, pady=3)
        calib_card.pack(fill=tk.X, padx=10, pady=2)

        tk.Button(calib_card, text="📐 1-Click: Full Arena View", font=("Segoe UI", 8, "bold"),
                  bg="#1c2536", fg="#00e5ff", command=self.apply_full_view_preset).pack(fill=tk.X, pady=1)

        tk.Button(calib_card, text="🎯 Interactive Crop & Warp Arena", font=("Segoe UI", 8, "bold"),
                  bg="#1e2235", fg="#00ffcc", command=self.start_calibration).pack(fill=tk.X, pady=1)

        # Live Ground Truth Telemetry
        tele_card = tk.LabelFrame(panel, text="LIVE ARENA TELEMETRY", font=("Segoe UI", 8, "bold"),
                                  fg="#64ffda", bg="#161926", bd=1, padx=8, pady=3)
        tele_card.pack(fill=tk.X, padx=10, pady=2)

        def make_row(parent, title):
            row = tk.Frame(parent, bg="#161926")
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=title, font=("Segoe UI", 8), fg="#8892b0", bg="#161926").pack(side=tk.LEFT)
            val = tk.Label(row, text="--", font=("Consolas", 8, "bold"), fg="#ffffff", bg="#161926")
            val.pack(side=tk.RIGHT)
            return val

        self.val_target = make_row(tele_card, "Auto Weak-Spot:")
        self.val_danger = make_row(tele_card, "Enemy Weapon Arc:")
        self.val_my_pos = make_row(tele_card, "Our Robot (X, Y):")
        self.val_my_hdg = make_row(tele_card, "Our Heading (θ):")
        self.val_enemy_pos = make_row(tele_card, "Enemy Robot (X, Y):")
        self.val_enemy_hdg = make_row(tele_card, "Enemy Heading (θ):")
        self.val_distance = make_row(tele_card, "Target Distance:")
        self.val_err = make_row(tele_card, "Steering Error:")
        self.val_pwm = make_row(tele_card, "Motors (L/R PWM):")
        self.val_user_throttle = make_row(tele_card, "Your Throttle Input:")

        # Serial / ESP-NOW Output Stream
        serial_card = tk.LabelFrame(panel, text="ESP-NOW WIRELESS PACKET", font=("Segoe UI", 8, "bold"),
                                    fg="#64ffda", bg="#161926", bd=1, padx=8, pady=2)
        serial_card.pack(fill=tk.X, padx=10, pady=2)

        self.lbl_serial_port = tk.Label(serial_card, text=f"Port: {self.serial_cfg.PORT} | ONLINE",
                                        font=("Segoe UI", 8), fg="#00e5ff", bg="#161926")
        self.lbl_serial_port.pack(anchor=tk.W)

        self.lbl_packet = tk.Label(serial_card, text="<SEARCH,0,0,0,0>", font=("Consolas", 8, "bold"),
                                   fg="#00ff88", bg="#0d0e14", padx=4, pady=2, relief=tk.SUNKEN)
        self.lbl_packet.pack(fill=tk.X, pady=1)

        # ==========================================
        # TAB 2: AI COMBAT CO-PILOT & COACH VIEW
        # ==========================================
        self.frame_tab_ai = tk.Frame(self.container, bg="#0b0c10")
        # Hidden initially; shown when Tab 2 button is clicked

        # Top Prominent AI Coaching Banner
        coach_banner = tk.Frame(self.frame_tab_ai, bg="#161b26", bd=2, relief=tk.GROOVE)
        coach_banner.pack(fill=tk.X, padx=12, pady=10)

        tk.Label(coach_banner, text="🤖 AI REAL-TIME DRIVING COACH & TACTICAL ADVISOR",
                 font=("Segoe UI", 11, "bold"), fg="#00e5ff", bg="#161b26").pack(anchor=tk.W, padx=12, pady=(8, 2))

        self.lbl_ai_tip = tk.Label(coach_banner, text="Searching arena... AI is scanning for enemy robot weak spots.",
                                   font=("Segoe UI", 14, "bold"), fg="#ffb703", bg="#161b26", wraplength=1300, justify=tk.LEFT)
        self.lbl_ai_tip.pack(anchor=tk.W, padx=12, pady=(2, 10))

        # Split Body in Tab 2: Left = Drive Assist Controls & Diagnostics; Right = AI Event Terminal
        ai_split = tk.Frame(self.frame_tab_ai, bg="#0b0c10")
        ai_split.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        ai_left = tk.Frame(ai_split, bg="#11131c", width=540, bd=1, relief=tk.SOLID)
        ai_left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

        ai_right = tk.Frame(ai_split, bg="#11131c", bd=1, relief=tk.SOLID)
        ai_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Left Column: How to Drive Card
        guide_card = tk.LabelFrame(ai_left, text="HOW TO DRIVE WITH AI STEER-BY-WIRE", font=("Segoe UI", 9, "bold"),
                                   fg="#64ffda", bg="#161926", bd=1, padx=12, pady=8)
        guide_card.pack(fill=tk.X, padx=10, pady=8)

        instructions = (
            "1. YOU CONTROL FRONT & BACK:\n"
            "   • Hold [W] or [↑]: Drive Forward with Power\n"
            "   • Hold [S] or [↓]: Reverse / Back away from danger\n"
            "   • Press [SPACE]: Instant Brake / Neutral\n\n"
            "2. AI CONTROLS LEFT & RIGHT:\n"
            "   • The AI continuously aims your robot dead-center at the enemy's weak spot.\n"
            "   • When you press [W], you charge directly into their rear motors or wheels!\n\n"
            "3. WEAPON EVASION:\n"
            "   • When the enemy's weapon faces us, AI automatically curves away from the danger zone."
        )
        tk.Label(guide_card, text=instructions, font=("Consolas", 9), fg="#ccd6f6", bg="#161926", justify=tk.LEFT).pack(anchor=tk.W)

        # Throttle Power Slider
        slider_card = tk.LabelFrame(ai_left, text="THROTTLE POWER SETTING", font=("Segoe UI", 9, "bold"),
                                    fg="#00e5ff", bg="#161926", bd=1, padx=12, pady=8)
        slider_card.pack(fill=tk.X, padx=10, pady=8)

        self.slider_pwr = tk.Scale(slider_card, from_=100, to=255, orient=tk.HORIZONTAL,
                                   bg="#161926", fg="#00ffcc", highlightthickness=0,
                                   command=self.on_slider_change)
        self.slider_pwr.set(220)
        self.slider_pwr.pack(fill=tk.X, pady=4)

        # System Health & Diagnostic Monitor Card
        health_card = tk.LabelFrame(ai_left, text="HARDWARE & VISION DIAGNOSTICS", font=("Segoe UI", 9, "bold"),
                                    fg="#ffaa00", bg="#161926", bd=1, padx=12, pady=8)
        health_card.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.lbl_diag_cam = make_row(health_card, "Camera 1 (Local):")
        self.lbl_diag_net = make_row(health_card, "Camera 2 (Network):")
        self.lbl_diag_track = make_row(health_card, "Visual Template Match:")
        self.lbl_diag_enemy = make_row(health_card, "Enemy Classification:")
        self.lbl_diag_serial = make_row(health_card, "ESP32 USB Serial:")

        # Right Column: Live AI Diagnostic Terminal
        tk.Label(ai_right, text="LIVE AI TACTICAL DIAGNOSTIC STREAM", font=("Segoe UI", 9, "bold"),
                 fg="#00e5ff", bg="#11131c").pack(anchor=tk.W, padx=10, pady=(8, 4))

        self.txt_ai_log = tk.Text(ai_right, bg="#07080b", fg="#00ff88", font=("Consolas", 9),
                                  bd=0, padx=8, pady=8, state=tk.DISABLED)
        self.txt_ai_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.refresh_thumbnail()

    def switch_tab(self, tab_name: str):
        if tab_name == "ARENA":
            self.frame_tab_ai.pack_forget()
            self.frame_tab_arena.pack(fill=tk.BOTH, expand=True)
            self.btn_tab_arena.config(bg="#00e5ff", fg="#000000")
            self.btn_tab_ai.config(bg="#1c2536", fg="#00e5ff")
        else:
            self.frame_tab_arena.pack_forget()
            self.frame_tab_ai.pack(fill=tk.BOTH, expand=True)
            self.btn_tab_ai.config(bg="#00e5ff", fg="#000000")
            self.btn_tab_arena.config(bg="#1c2536", fg="#00e5ff")

    def log_ai_message(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}\n"
        if len(self.ai_log_entries) == 0 or self.ai_log_entries[-1][9:] != msg + "\n":
            self.ai_log_entries.append(entry)
            if len(self.ai_log_entries) > 100:
                self.ai_log_entries.pop(0)
            self.txt_ai_log.config(state=tk.NORMAL)
            self.txt_ai_log.insert(tk.END, entry)
            self.txt_ai_log.see(tk.END)
            self.txt_ai_log.config(state=tk.DISABLED)

    def on_slider_change(self, val):
        self.user_throttle_power = int(val)
        if self.key_fwd_held:
            self.user_throttle_active = self.user_throttle_power
        elif self.key_rev_held:
            self.user_throttle_active = -int(self.user_throttle_power * 0.85)

    def init_camera(self):
        # Crash-proof CameraStream with watchdog
        self.cam = CameraStream(0, self.cam_cfg.FRAME_WIDTH, self.cam_cfg.FRAME_HEIGHT, self.cam_cfg.FPS, "Corner-A")

    def refresh_thumbnail(self):
        path = "my_robot_template.png"
        if os.path.exists(path):
            try:
                img = Image.open(path).resize((50, 50))
                p = ImageTk.PhotoImage(img)
                self.lbl_bot_thumb.config(image=p)
                self.lbl_bot_thumb.image = p
            except Exception:
                pass

    def upload_bot_photo(self):
        file_path = filedialog.askopenfilename(title="Select Picture of Our Combat Robot",
                                               filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.tracker.upload_my_template_file(file_path)
            self.refresh_thumbnail()
            messagebox.showinfo("Uploaded", "Our Robot template photo updated successfully!")
            self.log_ai_message("New bot photo template uploaded and multi-angle pyramid compiled.")

    def crop_bot_photo(self):
        if not self.cam:
            return
        ret, frame = self.cam.read()
        if ret and frame is not None:
            warped = self.calibrator.warp(frame, cam_num=1) if self.calibrator.H_cam1 is not None else frame
            roi = cv2.selectROI("Crop Our Robot (Space/Enter to Confirm)", warped, showCrosshair=True)
            cv2.destroyWindow("Crop Our Robot (Space/Enter to Confirm)")
            if roi[2] > 10 and roi[3] > 10:
                self.tracker.set_my_template_from_roi(warped, roi)
                self.refresh_thumbnail()
                self.log_ai_message("Bot template cropped from live arena view.")

    def apply_full_view_preset(self):
        if not self.cam:
            return
        ret, frame = self.cam.read()
        if ret and frame is not None:
            self.calibrator.set_full_frame_preset(1, frame.shape[1], frame.shape[0])
            messagebox.showinfo("Calibrated", "1-Click Full Arena Preset applied!")
            self.log_ai_message("Full arena view homography preset applied.")

    def start_calibration(self):
        if not self.cam:
            messagebox.showerror("Camera Offline", "Local camera is not connected.")
            return
        ret, frame = self.cam.read()
        if not ret or frame is None:
            messagebox.showerror("Camera Offline", "Unable to read frame from camera.")
            return

        self.is_paused = True
        try:
            success = self.calibrator.calibrate_camera(1, frame=frame, cam_source=self.cam)
            if success:
                messagebox.showinfo("Arena Calibrated", "Arena perspective crop & warp saved successfully!")
                self.log_ai_message("Interactive 4-corner perspective crop & warp saved.")
        except Exception as e:
            messagebox.showerror("Calibration Error", f"Error during calibration: {e}")
        finally:
            self.is_paused = False

    def toggle_estop(self):
        self.estop_active = not self.estop_active
        if self.estop_active:
            self.btn_estop.config(text="⚠️ E-STOP ACTIVE (CLICK TO RESUME)", bg="#ff9f1c", fg="#000000")
            self.lbl_action.config(text="E-STOP ACTIVE", fg="#e60026")
            self.log_ai_message("EMERGENCY STOP TRIGGERED! Motors locked.")
            if self.serial_bridge.is_connected:
                self.serial_bridge.send_packet("<STOP,0,0,0,0>\n")
        else:
            self.btn_estop.config(text="🛑 EMERGENCY STOP (E-STOP)", bg="#e60026", fg="#ffffff")
            self.log_ai_message("E-STOP Resumed. Autonomous steering re-engaged.")

    def update_loop(self):
        if not self.is_running:
            return

        if not self.is_paused and self.cam is not None:
            ret1, frame1 = self.cam.read()
            if frame1 is not None:
                if self.calibrator.H_cam1 is None:
                    self.calibrator.set_default_camera_corners(1, frame1.shape[1], frame1.shape[0])

                # 1. Local Warp & Visual Tracking
                warped1 = self.calibrator.warp(frame1, cam_num=1)
                if warped1 is None:
                    warped1 = cv2.resize(frame1, (self.arena_cfg.WARP_WIDTH, self.arena_cfg.WARP_HEIGHT))
                local_track = self.tracker.track(warped1)

                # 2. Receive Telemetry + Live Video from Laptop 2
                remote_data, remote_frame, remote_fresh, sender_ip = self.udp_receiver.get_latest()

                if sender_ip and self.udp_sender is None:
                    self.udp_sender = UDPCornerSender(sender_ip, target_port=self.net_cfg.UDP_PORT)

                if self.udp_sender is not None:
                    self.udp_sender.send_frame_and_telemetry(local_track, frame_bgr=frame1, estop=self.estop_active)

                if remote_fresh and remote_data and remote_data.get("estop", False):
                    if not self.estop_active:
                        self.toggle_estop()

                # Update Network Link Indicator
                if remote_fresh:
                    self.lbl_net_status.config(text=f"● LAPTOP 2 LINKED ({sender_ip})", fg="#00ff88")
                else:
                    self.lbl_net_status.config(text="● LAPTOP 2: WAITING FOR LINK", fg="#ffaa00")

                # 3. Dual-Corner Fusion
                fused_my, fused_enemy = self.fuse(local_track, remote_data, remote_fresh)

                # 4. Determine Drive Throttle (Steer-by-Wire vs Full Auto)
                if self.control_mode.get() == "STEER_ASSIST":
                    effective_throttle = self.user_throttle_active
                else:
                    effective_throttle = None

                # 5. Steering Math with Autonomous Weak-Spot Discovery
                steering = self.navigator.compute_steering(
                    my_pos=fused_my["pos"],
                    my_heading_deg=fused_my["heading"],
                    enemy_data=fused_enemy,
                    tactic_mode="SMART_FLANK",  # AI automatically decides best weak-spot
                    user_throttle=effective_throttle
                )

                # 6. Serial ESP-NOW Transmission
                packet = "<STOP,0,0,0,0>\n" if self.estop_active else steering["packet"]
                if self.serial_bridge.is_connected:
                    self.serial_bridge.send_packet(packet)
                    self.lbl_serial_badge.config(text="● ESP-NOW: BROADCASTING", fg="#00ff88")
                else:
                    self.lbl_serial_badge.config(text="● ESP-NOW: MOCK / READY", fg="#00e5ff")

                # FPS
                self.frame_count += 1
                now = time.time()
                if now - self.fps_time >= 1.0:
                    self.fps = self.frame_count / (now - self.fps_time)
                    self.frame_count = 0
                    self.fps_time = now
                    self.lbl_fps.config(text=f"{self.fps:.1f} FPS")

                # 7. Render Central Visual Arena Screen
                disp_arena = warped1.copy()
                self.draw_arena_overlays(disp_arena, fused_my, fused_enemy, steering)

                cam1_tile = self.build_camera_tile(frame1, "CAM 1: CORNER A (LOCAL LIVE)", is_local=True)
                cam2_tile = self.build_camera_tile(remote_frame if remote_fresh else None, "CAM 2: CORNER B (STREAMED)", is_local=False)

                right_stack = np.vstack([cam1_tile, cam2_tile])
                composite = np.hstack([disp_arena, right_stack])

                self.render_to_canvas(composite)
                self.update_telemetry_ui(fused_my, fused_enemy, steering)

        self.root.after(25, self.update_loop)

    def fuse(self, local_track, remote_data, remote_fresh):
        my_local = local_track["my_bot"]
        enemy_local = local_track["enemy_bot"]

        my_rem = remote_data.get("my_bot", {}) if (remote_fresh and remote_data) else {}
        enemy_rem = remote_data.get("enemy_bot", {}) if (remote_fresh and remote_data) else {}

        if my_local.get("detected"):
            fused_my = dict(my_local)
            fused_my["source"] = "CAM 1 (LOCAL)"
        elif remote_fresh and my_rem.get("detected"):
            fused_my = dict(my_rem)
            fused_my["source"] = "LAPTOP 2 (FALLBACK)"
        else:
            fused_my = {"detected": False, "pos": None, "heading": None, "source": "NONE"}

        if enemy_local.get("detected"):
            fused_enemy = dict(enemy_local)
            fused_enemy["source"] = "CAM 1 (LOCAL)"
        elif remote_fresh and enemy_rem.get("detected"):
            fused_enemy = dict(enemy_rem)
            fused_enemy["source"] = "LAPTOP 2 (FALLBACK)"
        else:
            fused_enemy = {"detected": False, "pos": None, "source": "NONE"}

        return fused_my, fused_enemy

    def build_camera_tile(self, frame_bgr, label, is_local=True):
        tile_w, tile_h = 320, 400
        if frame_bgr is None:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            cv2.rectangle(tile, (0, 0), (tile_w, tile_h), (25, 27, 36), -1)
            cv2.putText(tile, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 150), 1)
            cv2.putText(tile, "CONNECTING...", (85, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 170, 0), 1)
            return tile
            
        scaled = cv2.resize(frame_bgr, (tile_w, tile_h))
        cv2.rectangle(scaled, (0, 0), (tile_w, 36), (10, 12, 18), -1)
        color = (0, 255, 200) if is_local else (255, 200, 0)
        cv2.putText(scaled, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return scaled

    def draw_arena_overlays(self, canvas, my_bot, enemy_bot, steering):
        """Draws rich tactical overlay. Protected against any OpenCV invalid parameter exceptions."""
        try:
            err = steering.get("heading_error", 0.0)
            dist = steering.get("distance", 0.0)

            # 1. Our Bot (Cyan / Green Circle with Heading Arrow)
            if my_bot.get("detected") and my_bot.get("pos") is not None:
                mx, my = int(round(my_bot["pos"][0])), int(round(my_bot["pos"][1]))
                cv2.circle(canvas, (mx, my), 28, (0, 255, 120), 2, cv2.LINE_AA)
                cv2.circle(canvas, (mx, my), 4, (0, 255, 255), -1, cv2.LINE_AA)
                hdg = my_bot.get("heading")
                if hdg is not None:
                    rad = np.radians(hdg)
                    ax = int(round(mx + 45 * np.cos(rad)))
                    ay = int(round(my + 45 * np.sin(rad)))
                    cv2.arrowedLine(canvas, (mx, my), (ax, ay), (0, 255, 255), 3, tipLength=0.3)
                cv2.putText(canvas, f"OUR BOT [{my_bot.get('source', 'CAM')}]", (mx - 50, my - 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

            # 2. Enemy Bot (Oriented Chassis + Weapon Danger Cone + Weak Spots)
            if enemy_bot.get("detected") and enemy_bot.get("pos") is not None:
                ex, ey = int(round(enemy_bot["pos"][0])), int(round(enemy_bot["pos"][1]))
                en_hdg = enemy_bot.get("heading")

                # Oriented Bounding Box
                box_pts = enemy_bot.get("box_points")
                if box_pts and len(box_pts) == 4:
                    b_arr = np.array(box_pts, dtype=np.int32)
                    cv2.polylines(canvas, [b_arr], isClosed=True, color=(0, 60, 255), thickness=2, lineType=cv2.LINE_AA)
                    overlay = canvas.copy()
                    cv2.fillPoly(overlay, [b_arr], (0, 30, 180))
                    cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)
                else:
                    cv2.circle(canvas, (ex, ey), 30, (0, 0, 255), 2, cv2.LINE_AA)

                # Front Weapon Danger Arc / Cone
                if en_hdg is not None:
                    rad_e = np.radians(en_hdg)
                    fax = int(round(ex + 48 * np.cos(rad_e)))
                    fay = int(round(ey + 48 * np.sin(rad_e)))
                    cv2.arrowedLine(canvas, (ex, ey), (fax, fay), (0, 140, 255), 2, tipLength=0.35)

                    overlay = canvas.copy()
                    start_a = int(round(en_hdg - 65))
                    end_a = int(round(en_hdg + 65))
                    cv2.ellipse(overlay, (ex, ey), (95, 95), 0, start_a, end_a, (0, 0, 220), -1)
                    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
                    cv2.ellipse(canvas, (ex, ey), (95, 95), 0, start_a, end_a, (0, 0, 255), 1, cv2.LINE_AA)

                    lbl_px = int(round(ex + 70 * np.cos(rad_e)))
                    lbl_py = int(round(ey + 70 * np.sin(rad_e)))
                    cv2.putText(canvas, "! WEAPON ZONE !", (lbl_px - 45, lbl_py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 50, 255), 1, cv2.LINE_AA)

                # Render Weak Spots on Enemy
                weak_spots = enemy_bot.get("weak_spots", {})
                if weak_spots:
                    if weak_spots.get("rear") is not None:
                        rx, ry = int(round(weak_spots["rear"][0])), int(round(weak_spots["rear"][1]))
                        cv2.circle(canvas, (rx, ry), 10, (0, 165, 255), 2, cv2.LINE_AA)
                        cv2.circle(canvas, (rx, ry), 3, (0, 200, 255), -1, cv2.LINE_AA)
                        cv2.putText(canvas, "WEAK: REAR", (rx - 35, ry + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1, cv2.LINE_AA)

                    if weak_spots.get("left_wheel") is not None:
                        lx, ly = int(round(weak_spots["left_wheel"][0])), int(round(weak_spots["left_wheel"][1]))
                        cv2.circle(canvas, (lx, ly), 8, (255, 220, 0), 2, cv2.LINE_AA)
                        cv2.putText(canvas, "L-WHEEL", (lx - 25, ly - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1, cv2.LINE_AA)

                    if weak_spots.get("right_wheel") is not None:
                        rx2, ry2 = int(round(weak_spots["right_wheel"][0])), int(round(weak_spots["right_wheel"][1]))
                        cv2.circle(canvas, (rx2, ry2), 8, (255, 220, 0), 2, cv2.LINE_AA)
                        cv2.putText(canvas, "R-WHEEL", (rx2 - 25, ry2 + 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1, cv2.LINE_AA)

                cv2.putText(canvas, f"ENEMY BOT [{enemy_bot.get('source', 'CAM')}]", (ex - 50, ey - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA)

            # 3. Active Target Reticle & Steering Trajectory
            if my_bot.get("detected") and enemy_bot.get("detected"):
                mx, my = int(round(my_bot["pos"][0])), int(round(my_bot["pos"][1]))
                target_pt = steering.get("target_pos", enemy_bot["pos"])
                if target_pt is not None:
                    tx, ty = int(round(target_pt[0])), int(round(target_pt[1]))
                    cv2.circle(canvas, (tx, ty), 16, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.drawMarker(canvas, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)
                    t_name = steering.get("target_name", "TARGET")
                    cv2.putText(canvas, f"AUTO-LOCK: {t_name}", (tx + 18, ty - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

                waypoint = steering.get("flanking_waypoint")
                if waypoint is not None:
                    wx, wy = int(round(waypoint[0])), int(round(waypoint[1]))
                    cv2.circle(canvas, (wx, wy), 9, (0, 200, 255), -1, cv2.LINE_AA)
                    cv2.putText(canvas, "DODGE WEAPON WAYPOINT", (wx + 12, wy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)
                    cv2.line(canvas, (mx, my), (wx, wy), (0, 200, 255), 2, cv2.LINE_AA)
                    if target_pt is not None:
                        cv2.line(canvas, (wx, wy), (tx, ty), (0, 255, 120), 2, cv2.LINE_AA)
                elif target_pt is not None:
                    col = (0, 255, 0) if abs(err) <= 20 else ((0, 255, 255) if abs(err) <= 60 else (0, 165, 255))
                    cv2.line(canvas, (mx, my), (tx, ty), col, 2, cv2.LINE_AA)

                # HUD Top banner showing real-time AI Driving Coach tip right on arena
                tip = steering.get("ai_tip", "")
                cv2.rectangle(canvas, (0, 0), (self.arena_cfg.WARP_WIDTH, 38), (14, 16, 24), -1)
                cv2.putText(canvas, f"AI CO-PILOT: {tip}", (14, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 220), 1, cv2.LINE_AA)
        except Exception as e:
            pass

    def render_to_canvas(self, bgr_img):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        scale = min(cw / im.width, ch / im.height)
        new_w, new_h = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        im_res = im.resize((new_w, new_h), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(image=im_res)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor=tk.CENTER)

    def update_telemetry_ui(self, my_bot, enemy_bot, steering):
        cmd = "E-STOP" if self.estop_active else steering.get("command", "SEARCH")
        cmd_col = {
            "PURSUIT": "#00ff88",
            "RAM_WEAK": "#ff0055",
            "CO_FWD": "#00ffcc",
            "CO_REV": "#ffbb00",
            "AIM_LOCK": "#00e5ff",
            "AIM_READY": "#00ff88",
            "FLANK_ARC": "#00e5ff",
            "SPIN_LEFT": "#ffcc00",
            "SPIN_RIGHT": "#ffcc00",
            "SEARCH": "#8892b0",
            "E-STOP": "#e60026"
        }
        self.lbl_action.config(text=cmd, fg=cmd_col.get(cmd, "#ffffff"))

        my_str = f"({int(my_bot['pos'][0])}, {int(my_bot['pos'][1])})" if my_bot.get("detected") else "LOST"
        hdg_str = f"{my_bot['heading']:.1f}°" if my_bot.get("heading") is not None else "--"
        en_str = f"({int(enemy_bot['pos'][0])}, {int(enemy_bot['pos'][1])})" if enemy_bot.get("detected") else "LOST"

        self.val_target.config(text=f"{steering.get('target_name', '--')} ({steering.get('tactic_phase', '')})")
        
        in_danger = steering.get("in_danger_cone", False)
        if in_danger:
            self.val_danger.config(text="⚠️ IN WEAPON ARC (FLANKING)", fg="#ff3355")
        else:
            self.val_danger.config(text="● CLEAR / SAFE ANGLE", fg="#00ff88")

        en_hdg = enemy_bot.get("heading")
        self.val_enemy_hdg.config(text=f"{en_hdg:.1f}°" if en_hdg is not None else "--")

        self.val_my_pos.config(text=my_str)
        self.val_my_hdg.config(text=hdg_str)
        self.val_enemy_pos.config(text=en_str)
        self.val_distance.config(text=f"{steering.get('distance', 0.0)} px")
        self.val_err.config(text=f"{steering.get('heading_error', 0.0):+.1f}°")
        self.val_pwm.config(text=f"{steering.get('throttle_left', 0)} / {steering.get('throttle_right', 0)}")
        
        # User Throttle indicator
        if self.user_throttle_active > 0:
            th_txt = f"▲ FORWARD (+{self.user_throttle_active} PWM)"
            th_fg = "#00ff88"
        elif self.user_throttle_active < 0:
            th_txt = f"▼ REVERSE ({self.user_throttle_active} PWM)"
            th_fg = "#ffaa00"
        else:
            th_txt = "● NEUTRAL (0 PWM) [AI AIMING]"
            th_fg = "#8892b0"
        self.val_user_throttle.config(text=th_txt, fg=th_fg)

        if self.estop_active:
            self.lbl_packet.config(text="<STOP,0,0,0,0>", fg="#e60026")
        else:
            self.lbl_packet.config(text=steering.get("packet", "<SEARCH,0,0,0,0>").strip(), fg="#00ff88")

        # Update Tab 2: AI Coach Banner & Diagnostic Terminal
        tip = steering.get("ai_tip", "")
        alert = steering.get("ai_alert", "INFO")
        tip_colors = {
            "DANGER": "#ff3355",
            "RAM_NOW": "#ff0055",
            "ATTACK": "#00ffcc",
            "ALIGNING": "#ffbb00",
            "LOCKED": "#00ff88",
            "SEARCHING": "#8892b0"
        }
        self.lbl_ai_tip.config(text=tip, fg=tip_colors.get(alert, "#ffffff"))

        # Update Tab 2: Hardware Diagnostics
        cam_stat = self.cam.get_status() if self.cam is not None else {"status": "OFFLINE", "fps": 0}
        self.lbl_diag_cam.config(text=f"{cam_stat.get('status')} ({cam_stat.get('fps')} FPS)",
                                 fg="#00ff88" if cam_stat.get("is_opened") else "#ff4455")
        
        rem_data, _, rem_fresh, rem_ip = self.udp_receiver.get_latest()
        self.lbl_diag_net.config(text=f"Corner B ({rem_ip}) - LINKED" if rem_fresh else "WAITING FOR LAPTOP 2",
                                 fg="#00ff88" if rem_fresh else "#ffaa00")

        self.lbl_diag_track.config(text=f"Score: {my_bot.get('score', 0):.2f} (LOCKED)" if my_bot.get("detected") else "SEARCHING FOR TEMPLATE",
                                   fg="#00ff88" if my_bot.get("detected") else "#ffaa00")

        self.lbl_diag_enemy.config(text=f"Area: {enemy_bot.get('area', 0)}px | θ: {en_hdg}°" if enemy_bot.get("detected") else "NO ENEMY IN ARENA",
                                   fg="#00ff88" if enemy_bot.get("detected") else "#8892b0")

        self.lbl_diag_serial.config(text=f"{self.serial_cfg.PORT} (115200) - READY" if self.serial_bridge.is_connected else "MOCK / READY",
                                    fg="#00e5ff")

        # Log significant tactical changes
        if alert in ["DANGER", "RAM_NOW"]:
            self.log_ai_message(f"[{alert}] {tip}")

    def on_close(self):
        self.is_running = False
        if self.cam is not None:
            self.cam.stop()
        self.serial_bridge.close()
        self.udp_receiver.close()
        if self.udp_sender is not None:
            self.udp_sender.close()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = MasterCommandDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()

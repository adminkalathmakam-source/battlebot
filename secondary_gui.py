"""
SECONDARY LAPTOP TACTICAL COMMAND DASHBOARD (ASUS TUF 2 - Corner B).
Professional Esports / Defense Grade Combat Robotics Ground-Truth Visualizer.
Features:
- Dual-Camera Live Feeds: Displays BOTH Corner B (Local) and Corner A (Streamed from Master)
- Photo Upload & Live Cropper for OUR ROBOT ONLY (Enemy is automatically any other object)
- 1-Click Arena Calibration Presets (Full View / Centered Box / Interactive 4-Point)
- Real-Time UDP Telemetry & Video Streamer to Master Laptop
- Remote E-STOP Broadcast Kill Switch
"""

import os
import sys
import time
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

from config import ArenaConfig, CameraConfig, VisualTrackingConfig, NetworkConfig
from camera_stream import CameraStream
from homography import ArenaCalibrator
from visual_tracker import VisualRobotTracker
from network_bridge import UDPCornerSender, UDPCornerReceiver

class SecondaryCommandDashboard:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("COMBAT ROBOTICS COMMAND SYSTEM - SECONDARY NODE (ASUS TUF 2)")
        self.root.geometry("1420x900")
        self.root.configure(bg="#0b0c10")

        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        # Configs
        self.arena_cfg = ArenaConfig()
        self.cam_cfg = CameraConfig()
        self.track_cfg = VisualTrackingConfig()
        self.net_cfg = NetworkConfig()

        self.calibrator = ArenaCalibrator(self.arena_cfg.WARP_WIDTH, self.arena_cfg.WARP_HEIGHT, "calibration_secondary.json")
        self.tracker = VisualRobotTracker(self.track_cfg, self.arena_cfg)
        
        # Dual-Video UDP Network Bridge
        self.udp_sender = UDPCornerSender(self.net_cfg.MASTER_IP, target_port=self.net_cfg.UDP_PORT)
        self.udp_receiver = UDPCornerReceiver(listen_port=self.net_cfg.UDP_PORT)

        self.cam = None
        self.is_running = True
        self.is_paused = False
        self.estop_active = False

        self.calibrating = False
        self.calib_clicks = []

        self.setup_ui()
        self.init_camera()

        self.fps_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.update_loop()

    def setup_ui(self):
        # 1. Top Status Banner
        header = tk.Frame(self.root, bg="#13151f", height=54, bd=0)
        header.pack(fill=tk.X, side=tk.TOP)

        title_frame = tk.Frame(header, bg="#13151f")
        title_frame.pack(side=tk.LEFT, padx=16, pady=8)

        tk.Label(title_frame, text="APEX COMBAT GROUND TRUTH", font=("Impact", 14),
                 fg="#ffb703", bg="#13151f").pack(side=tk.LEFT)
        tk.Label(title_frame, text="  [SECONDARY NODE - CORNER B]", font=("Segoe UI", 10, "bold"),
                 fg="#8892b0", bg="#13151f").pack(side=tk.LEFT)

        self.lbl_fps = tk.Label(header, text="30.0 FPS", font=("Consolas", 10, "bold"), fg="#00ff88", bg="#13151f")
        self.lbl_fps.pack(side=tk.RIGHT, padx=16)

        self.lbl_net_status = tk.Label(header, text=f"STREAMING TO: {self.net_cfg.MASTER_IP}:{self.net_cfg.UDP_PORT}",
                                       font=("Segoe UI", 9, "bold"), fg="#00e5ff", bg="#13151f")
        self.lbl_net_status.pack(side=tk.RIGHT, padx=14)

        # 2. Main Work Area
        body = tk.Frame(self.root, bg="#0b0c10")
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Video Display Canvas (Left Side)
        video_frame = tk.Frame(body, bg="#141622", bd=1, relief=tk.SOLID)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(video_frame, bg="#050608", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Control Panel (Right Side)
        panel = tk.Frame(body, bg="#11131c", width=360)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        # Emergency Stop Button (Broadcasts E-STOP over UDP to Master)
        self.btn_estop = tk.Button(panel, text="🛑 EMERGENCY STOP (E-STOP)", font=("Segoe UI", 13, "bold"),
                                   bg="#e60026", fg="#ffffff", activebackground="#ff1a3c",
                                   relief=tk.FLAT, pady=9, cursor="hand2", command=self.toggle_estop)
        self.btn_estop.pack(fill=tk.X, padx=10, pady=(10, 6))

        # Master Destination IP Card
        net_card = tk.LabelFrame(panel, text="MASTER LAPTOP CONNECTION", font=("Segoe UI", 8, "bold"),
                                 fg="#00e5ff", bg="#161926", bd=1, padx=8, pady=4)
        net_card.pack(fill=tk.X, padx=10, pady=3)

        ip_row = tk.Frame(net_card, bg="#161926")
        ip_row.pack(fill=tk.X, pady=2)
        tk.Label(ip_row, text="Master IP:", font=("Segoe UI", 8, "bold"), fg="#8892b0", bg="#161926").pack(side=tk.LEFT)
        self.entry_ip = tk.Entry(ip_row, font=("Consolas", 9), bg="#0d0e14", fg="#00e5ff", insertbackground="#ffffff")
        self.entry_ip.insert(0, self.net_cfg.MASTER_IP)
        self.entry_ip.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(6, 0))

        tk.Button(net_card, text="🔗 Connect to Master IP", font=("Segoe UI", 8, "bold"),
                  bg="#1f2937", fg="#00e5ff", command=self.update_master_ip).pack(fill=tk.X, pady=3)

        # Robot Profile (OUR BOT ONLY)
        bot_card = tk.LabelFrame(panel, text="OUR COMBAT ROBOT PROFILE", font=("Segoe UI", 8, "bold"),
                                 fg="#00e5ff", bg="#161926", bd=1, padx=8, pady=4)
        bot_card.pack(fill=tk.X, padx=10, pady=3)

        thumb_row = tk.Frame(bot_card, bg="#161926")
        thumb_row.pack(fill=tk.X, pady=2)

        self.lbl_bot_thumb = tk.Label(thumb_row, bg="#090a0f", width=64, height=64, bd=1, relief=tk.SOLID)
        self.lbl_bot_thumb.pack(side=tk.LEFT, padx=4)

        desc_col = tk.Frame(thumb_row, bg="#161926")
        desc_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        tk.Label(desc_col, text="Template Match Active", font=("Segoe UI", 8, "bold"), fg="#00ff88", bg="#161926").pack(anchor=tk.W)
        tk.Label(desc_col, text="All other blobs = Enemy", font=("Segoe UI", 8), fg="#8892b0", bg="#161926").pack(anchor=tk.W)

        btn_row = tk.Frame(bot_card, bg="#161926")
        btn_row.pack(fill=tk.X, pady=4)

        tk.Button(btn_row, text="📁 Upload Photo...", font=("Segoe UI", 8, "bold"),
                  bg="#1f2937", fg="#00e5ff", command=self.upload_bot_photo).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        tk.Button(btn_row, text="✂️ Crop Live", font=("Segoe UI", 8, "bold"),
                  bg="#1f2937", fg="#ffffff", command=self.crop_bot_photo).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        # Arena Calibration Card (1-Click Presets)
        calib_card = tk.LabelFrame(panel, text="ARENA CALIBRATION PRESETS", font=("Segoe UI", 8, "bold"),
                                   fg="#ffaa00", bg="#161926", bd=1, padx=8, pady=4)
        calib_card.pack(fill=tk.X, padx=10, pady=3)

        tk.Button(calib_card, text="📐 1-Click: Full Arena View (Wide)", font=("Segoe UI", 8, "bold"),
                  bg="#1c2536", fg="#00e5ff", command=self.apply_full_view_preset).pack(fill=tk.X, pady=2)

        tk.Button(calib_card, text="⏹️ 1-Click: Centered Box Arena (Square)", font=("Segoe UI", 8, "bold"),
                  bg="#2a2336", fg="#ffd166", command=self.apply_square_preset).pack(fill=tk.X, pady=2)

        tk.Button(calib_card, text="🎯 Interactive Crop & Warp Arena", font=("Segoe UI", 8, "bold"),
                  bg="#1e2235", fg="#00ffcc", command=self.start_calibration).pack(fill=tk.X, pady=2)


        # Local Telemetry
        tele_card = tk.LabelFrame(panel, text="LOCAL CORNER B TELEMETRY", font=("Segoe UI", 8, "bold"),
                                  fg="#64ffda", bg="#161926", bd=1, padx=8, pady=3)
        tele_card.pack(fill=tk.X, padx=10, pady=3)

        def make_row(parent, title):
            row = tk.Frame(parent, bg="#161926")
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=title, font=("Segoe UI", 8), fg="#8892b0", bg="#161926").pack(side=tk.LEFT)
            val = tk.Label(row, text="--", font=("Consolas", 8, "bold"), fg="#ffffff", bg="#161926")
            val.pack(side=tk.RIGHT)
            return val

        self.val_my_pos = make_row(tele_card, "Our Robot (X, Y):")
        self.val_my_hdg = make_row(tele_card, "Our Heading (θ):")
        self.val_enemy_pos = make_row(tele_card, "Enemy Robot (X, Y):")
        self.val_enemy_hdg = make_row(tele_card, "Enemy Heading (θ):")
        self.val_frames = make_row(tele_card, "Frames Transmitted:")

        self.refresh_thumbnail()


    def init_camera(self):
        self.cam = CameraStream(0, self.cam_cfg.FRAME_WIDTH, self.cam_cfg.FRAME_HEIGHT, self.cam_cfg.FPS, "Corner-B")

    def refresh_thumbnail(self):
        path = "my_robot_template.png"
        if os.path.exists(path):
            try:
                img = Image.open(path).resize((60, 60))
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

    def apply_full_view_preset(self):
        if not self.cam:
            return
        ret, frame = self.cam.read()
        if ret and frame is not None:
            self.calibrator.set_full_frame_preset(1, frame.shape[1], frame.shape[0])
            messagebox.showinfo("Calibrated", "1-Click Full Arena Preset applied!")

    def apply_square_preset(self):
        if not self.cam:
            return
        ret, frame = self.cam.read()
        if ret and frame is not None:
            self.calibrator.set_centered_square_preset(1, frame.shape[1], frame.shape[0])
            messagebox.showinfo("Calibrated", "1-Click Centered Square Preset applied!")

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
        except Exception as e:
            messagebox.showerror("Calibration Error", f"Error during calibration: {e}")
        finally:
            self.is_paused = False

    def on_canvas_click(self, event):
        pass


    def update_master_ip(self):
        new_ip = self.entry_ip.get().strip()
        if new_ip:
            self.net_cfg.MASTER_IP = new_ip
            self.udp_sender.close()
            self.udp_sender = UDPCornerSender(new_ip, target_port=self.net_cfg.UDP_PORT)
            self.lbl_net_status.config(text=f"STREAMING TO: {new_ip}:{self.net_cfg.UDP_PORT}")
            messagebox.showinfo("Connected", f"Now streaming to Master Laptop at {new_ip}!")

    def toggle_estop(self):
        self.estop_active = not self.estop_active
        if self.estop_active:
            self.btn_estop.config(text="⚠️ E-STOP ACTIVE (CLICK TO RESUME)", bg="#ff9f1c", fg="#000000")
            self.udp_sender.send_frame_and_telemetry({"my_bot": {}, "enemy_bot": {}}, estop=True)
        else:
            self.btn_estop.config(text="🛑 EMERGENCY STOP (E-STOP)", bg="#e60026", fg="#ffffff")

    def update_loop(self):
        if not self.is_running:
            return

        if not self.is_paused and self.cam is not None:
            ret2, frame2 = self.cam.read()
            if ret2 and frame2 is not None:
                if self.calibrator.H_cam1 is None:
                    self.calibrator.set_default_camera_corners(1, frame2.shape[1], frame2.shape[0])

                # 1. Local Warp & Visual Tracking
                warped2 = self.calibrator.warp(frame2, cam_num=1)
                local_track = self.tracker.track(warped2)

                # 2. Transmit Corner B Frame + Coordinates to Master
                self.udp_sender.send_frame_and_telemetry(local_track, frame_bgr=frame2, estop=self.estop_active)

                # 3. Receive Master Laptop Corner A Live Video Stream
                master_telemetry, master_frame, master_fresh, _ = self.udp_receiver.get_latest()

                # FPS
                self.frame_count += 1
                now = time.time()
                if now - self.fps_time >= 1.0:
                    self.fps = self.frame_count / (now - self.fps_time)
                    self.frame_count = 0
                    self.fps_time = now
                    self.lbl_fps.config(text=f"{self.fps:.1f} FPS")

                # 4. Compose Visual Screen:
                # [ Central 800x800 Warped Arena ] + [ Right Stack: Cam 1 (Streamed) & Cam 2 (Local) ]
                disp_arena = warped2.copy()
                self.draw_arena_overlays(disp_arena, local_track["my_bot"], local_track["enemy_bot"])

                cam1_tile = self.build_camera_tile(master_frame if master_fresh else None, "CAM 1: CORNER A (STREAMED)", is_local=False)
                cam2_tile = self.build_camera_tile(frame2, "CAM 2: CORNER B (LOCAL LIVE)", is_local=True)

                right_stack = np.vstack([cam1_tile, cam2_tile])
                composite = np.hstack([disp_arena, right_stack])

                self.render_to_canvas(composite)
                self.update_telemetry_ui(local_track["my_bot"], local_track["enemy_bot"])

        self.root.after(25, self.update_loop)

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

    def draw_arena_overlays(self, canvas, my_bot, enemy_bot):
        try:
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
                cv2.putText(canvas, f"OUR BOT ({my_bot.get('score', 0):.2f})", (mx - 50, my - 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

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

                # Front Weapon Danger Arc
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

                # Weak Spots
                weak_spots = enemy_bot.get("weak_spots", {})
                if weak_spots:
                    if weak_spots.get("rear") is not None:
                        rx, ry = int(round(weak_spots["rear"][0])), int(round(weak_spots["rear"][1]))
                        cv2.circle(canvas, (rx, ry), 10, (0, 165, 255), 2, cv2.LINE_AA)
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

                cv2.putText(canvas, "ENEMY BOT (CORNER B)", (ex - 50, ey - 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1, cv2.LINE_AA)
        except Exception:
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

    def update_telemetry_ui(self, my_bot, enemy_bot):
        my_str = f"({int(my_bot['pos'][0])}, {int(my_bot['pos'][1])})" if my_bot.get("detected") else "LOST"
        hdg_str = f"{my_bot['heading']:.1f}°" if my_bot.get("heading") is not None else "--"
        en_str = f"({int(enemy_bot['pos'][0])}, {int(enemy_bot['pos'][1])})" if enemy_bot.get("detected") else "LOST"
        en_hdg = enemy_bot.get("heading")

        self.val_my_pos.config(text=my_str)
        self.val_my_hdg.config(text=hdg_str)
        self.val_enemy_pos.config(text=en_str)
        self.val_enemy_hdg.config(text=f"{en_hdg:.1f}°" if en_hdg is not None else "--")
        self.val_frames.config(text=str(self.frame_count))


    def on_close(self):
        self.is_running = False
        if self.cam is not None:
            self.cam.stop()
        self.udp_sender.close()
        self.udp_receiver.close()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = SecondaryCommandDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()

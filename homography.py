"""
Arena Calibration and Homography Perspective Warping.
Flattens trapezoidal corner-camera views into standardized 2D top-down coordinates.
"""

import os
import json
import cv2
import numpy as np

class ArenaCalibrator:
    def __init__(self, warp_width: int = 800, warp_height: int = 800, calib_file: str = "calibration.json"):
        self.warp_width = warp_width
        self.warp_height = warp_height
        self.calib_file = calib_file
        
        # Standard destination points in warped top-down space:
        # [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        self.dst_pts = np.array([
            [0, 0],
            [self.warp_width - 1, 0],
            [self.warp_width - 1, self.warp_height - 1],
            [0, self.warp_height - 1]
        ], dtype=np.float32)
        
        # Homography matrices for Cam 1 & Cam 2
        self.H_cam1 = None
        self.H_cam2 = None
        self.pts_cam1 = []
        self.pts_cam2 = []
        
        # Load saved calibration if available
        self.load_calibration()

    def load_calibration(self) -> bool:
        """Loads 4-point corner calibration and computes homography matrices from JSON file."""
        if not os.path.exists(self.calib_file):
            print(f"[WARN] No calibration file '{self.calib_file}' found. Run calibration first.")
            return False
            
        try:
            with open(self.calib_file, 'r') as f:
                data = json.load(f)
                
            if "cam1" in data and len(data["cam1"]) == 4:
                self.pts_cam1 = data["cam1"]
                src = np.array(self.pts_cam1, dtype=np.float32)
                self.H_cam1, _ = cv2.findHomography(src, self.dst_pts)
                
            if "cam2" in data and len(data["cam2"]) == 4:
                self.pts_cam2 = data["cam2"]
                src = np.array(self.pts_cam2, dtype=np.float32)
                self.H_cam2, _ = cv2.findHomography(src, self.dst_pts)
                
            print(f"[OK] Calibration loaded successfully from {self.calib_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load calibration file: {e}")
            return False

    def set_default_camera_corners(self, cam_num: int, frame_w: int, frame_h: int):
        self.set_full_frame_preset(cam_num, frame_w, frame_h)

    def set_full_frame_preset(self, cam_num: int, frame_w: int, frame_h: int):
        """Preset 1: Instant full camera view with 4% margins."""
        margin_x = int(frame_w * 0.04)
        margin_y = int(frame_h * 0.04)
        pts = [
            [margin_x, margin_y],
            [frame_w - margin_x, margin_y],
            [frame_w - margin_x, frame_h - margin_y],
            [margin_x, frame_h - margin_y]
        ]
        self._apply_preset_pts(cam_num, pts)
        self.save_calibration()
        print(f"[OK] Applied FULL-VIEW preset to Cam {cam_num}.")

    def set_centered_square_preset(self, cam_num: int, frame_w: int, frame_h: int):
        """Preset 2: Instant centered square arena box."""
        box_side = int(min(frame_w, frame_h) * 0.85)
        cx, cy = frame_w // 2, frame_h // 2
        half = box_side // 2
        pts = [
            [cx - half, cy - half],  # TL
            [cx + half, cy - half],  # TR
            [cx + half, cy + half],  # BR
            [cx - half, cy + half]   # BL
        ]
        self._apply_preset_pts(cam_num, pts)
        self.save_calibration()
        print(f"[OK] Applied CENTERED-SQUARE preset to Cam {cam_num}.")

    def _apply_preset_pts(self, cam_num: int, pts: list):
        src = np.array(pts, dtype=np.float32)
        H, _ = cv2.findHomography(src, self.dst_pts)
        if cam_num == 1:
            self.pts_cam1 = pts
            self.H_cam1 = H
        else:
            self.pts_cam2 = pts
            self.H_cam2 = H


    def save_calibration(self):
        """Saves selected corner points to JSON."""
        data = {
            "warp_size": [self.warp_width, self.warp_height],
            "cam1": self.pts_cam1,
            "cam2": self.pts_cam2
        }
        with open(self.calib_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"[INFO] Calibration saved to {self.calib_file}")

    def interactive_calibrate(self, frame: np.ndarray, camera_name: str = "Camera 1") -> np.ndarray:
        """
        Interactive mouse-click calibration window.
        User clicks the 4 arena floor corners in clockwise order:
        1: Top-Left, 2: Top-Right, 3: Bottom-Right, 4: Bottom-Left.
        """
        clicked_pts = []
        display_frame = frame.copy()
        window_name = f"Calibrate {camera_name} - Click 4 Arena Corners (TL -> TR -> BR -> BL)"

        corner_labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]

        def mouse_cb(event, x, y, flags, param):
            nonlocal display_frame
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pts) < 4:
                clicked_pts.append([x, y])
                idx = len(clicked_pts) - 1
                
                # Draw marker and text
                cv2.circle(display_frame, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(display_frame, corner_labels[idx], (x + 10, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                # Connect lines if multiple points exist
                if len(clicked_pts) > 1:
                    cv2.line(display_frame, tuple(clicked_pts[-2]), tuple(clicked_pts[-1]), (0, 255, 0), 2)
                if len(clicked_pts) == 4:
                    cv2.line(display_frame, tuple(clicked_pts[3]), tuple(clicked_pts[0]), (0, 255, 0), 2)
                    
                cv2.imshow(window_name, display_frame)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_cb)

        # Instructions banner
        header = display_frame.copy()
        cv2.rectangle(header, (0, 0), (display_frame.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(header, f"CLICK 4 CORNERS: TL -> TR -> BR -> BL | Press 'r' to reset, 'ESC' to cancel",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        display_frame = header

        while True:
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF
            
            if len(clicked_pts) == 4:
                cv2.putText(display_frame, "Ready! Press [ENTER] or [SPACE] to accept", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(0) & 0xFF
                if key in [13, 32]:  # Enter or Space
                    break
                    
            if key == ord('r'):  # Reset
                clicked_pts = []
                display_frame = frame.copy()
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 50), (30, 30, 30), -1)
                cv2.putText(display_frame, f"CLICK 4 CORNERS: TL -> TR -> BR -> BL | Press 'r' to reset",
                            (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            elif key == 27:  # ESC
                print("[INFO] Calibration aborted.")
                cv2.destroyWindow(window_name)
                return None

        cv2.destroyWindow(window_name)
        
        src_pts = np.array(clicked_pts, dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, self.dst_pts)
        return H, clicked_pts

    def calibrate_camera(self, cam_num: int, frame: np.ndarray = None, cam_source = None) -> bool:
        """Calibrates cam 1 or cam 2 using interactive drag-and-crop visual warper."""
        from arena_cropper import InteractiveArenaWarper
        warper = InteractiveArenaWarper(self.warp_width, self.warp_height)
        existing_pts = self.pts_cam1 if cam_num == 1 else self.pts_cam2
        name = f"Camera {cam_num} - Drag 4 Corners to Fit Arena Floor"
        H, pts = warper.calibrate(frame=frame, cam_source=cam_source, existing_pts=existing_pts, window_title=name)
        if H is not None and pts is not None:
            if cam_num == 1:
                self.H_cam1 = H
                self.pts_cam1 = pts
            else:
                self.H_cam2 = H
                self.pts_cam2 = pts
            self.save_calibration()
            print(f"[OK] Camera {cam_num} calibrated and saved.")
            return True
        return False


    def warp(self, frame: np.ndarray, cam_num: int = 1) -> np.ndarray:
        """Warps raw camera perspective into top-down 2D arena floor view."""
        H = self.H_cam1 if cam_num == 1 else self.H_cam2
        if H is None or frame is None:
            return None
        return cv2.warpPerspective(frame, H, (self.warp_width, self.warp_height))

    @staticmethod
    def transform_point(point: tuple, H: np.ndarray) -> tuple:
        """Transforms a 2D (x, y) point from raw camera space into warped arena space."""
        if H is None or point is None:
            return None
        px = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(px, H)
        return float(transformed[0][0][0]), float(transformed[0][0][1])

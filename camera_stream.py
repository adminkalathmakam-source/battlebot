"""
Ultra-Robust, Auto-Reconnecting Camera Stream for Combat Robotics.
Features:
- Crash-proof background capture thread with automatic reconnection
- Multi-index fallback (auto-detects USB webcam index 0, 1, 2 or laptop webcam)
- DirectShow (CAP_DSHOW) with fallback to default backend (CAP_MSMF)
- Drop-frame watchdog: auto-resets if camera hangs or disconnects
- Safe fallback frame buffer: GUI never crashes even during disconnect
"""

import threading
import time
import cv2
import numpy as np

class CameraStream:
    def __init__(self, src: int = 0, width: int = 1280, height: int = 720, fps: int = 30, name: str = "Camera"):
        self.name = name
        self.preferred_src = src
        self.current_src = src
        self.width = width
        self.height = height
        self.fps = fps
        
        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.is_opened = False
        self.lock = threading.Lock()
        
        self.last_grab_time = time.time()
        self.status_message = "INITIALIZING"
        self.fps_actual = 0.0
        self.fps_counter = 0
        self.fps_timer = time.time()

        # Generate a safe standby image in case webcam takes a moment to connect
        self.standby_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.standby_frame[:] = (20, 24, 32)
        cv2.putText(self.standby_frame, f"{self.name.upper()} CONNECTING...", (self.width // 2 - 180, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

        self.cap = None
        self._open_camera()

        # Start background polling thread
        self.thread = threading.Thread(target=self._update_loop, daemon=True, name=f"{self.name}-Thread")
        self.thread.start()

    def _open_camera(self) -> bool:
        """Attempts to open preferred camera or fallbacks gracefully."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        # Try list of possible indices starting with requested
        candidate_indices = [self.preferred_src]
        for alt in [0, 1, 2]:
            if alt not in candidate_indices:
                candidate_indices.append(alt)

        for idx in candidate_indices:
            # 1. Try DirectShow (Preferred on Windows for zero-lag)
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    self.cap = cap
                    self.current_src = idx
                    self._configure_cap()
                    self.is_opened = True
                    self.status_message = f"ONLINE (Index {idx} - DirectShow)"
                    print(f"[OK] {self.name} connected on index {idx} via DirectShow.")
                    return True
            except Exception:
                pass

            # 2. Try default backend fallback
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    self.cap = cap
                    self.current_src = idx
                    self._configure_cap()
                    self.is_opened = True
                    self.status_message = f"ONLINE (Index {idx} - Default)"
                    print(f"[OK] {self.name} connected on index {idx} via Default backend.")
                    return True
            except Exception:
                pass

        self.is_opened = False
        self.status_message = "DISCONNECTED - RETRYING"
        print(f"[WARN] {self.name} could not open any video index. Watchdog will auto-retry.")
        return False

    def _configure_cap(self):
        """Sets resolution and FPS."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _update_loop(self):
        """Crash-proof continuous capture thread with watchdog."""
        while not self.stopped:
            try:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.8)
                    self._open_camera()
                    continue

                grabbed, frame = self.cap.read()
                now = time.time()

                if grabbed and frame is not None and frame.size > 0:
                    with self.lock:
                        self.frame = frame
                        self.grabbed = True
                        self.last_grab_time = now

                    self.fps_counter += 1
                    if now - self.fps_timer >= 1.0:
                        self.fps_actual = self.fps_counter / (now - self.fps_timer)
                        self.fps_counter = 0
                        self.fps_timer = now
                else:
                    # Frame drop check watchdog
                    if now - self.last_grab_time > 2.0:
                        self.status_message = "SIGNAL HANG - RECONNECTING"
                        time.sleep(0.5)
                        self._open_camera()
                    else:
                        time.sleep(0.005)

            except Exception as e:
                # Catch any unexpected camera driver error so thread never dies
                self.status_message = f"ERROR: {str(e)[:20]}"
                time.sleep(0.5)
                self._open_camera()

    def read(self):
        """Returns the freshest frame. Never raises exception or returns None."""
        with self.lock:
            if self.frame is not None and self.grabbed:
                return True, self.frame.copy()
            else:
                # Return standby placeholder frame so callers never crash
                return False, self.standby_frame.copy()

    def get_status(self) -> dict:
        """Returns camera health diagnostics."""
        return {
            "name": self.name,
            "status": self.status_message,
            "is_opened": self.is_opened,
            "index": self.current_src,
            "fps": round(self.fps_actual, 1),
            "healthy": (time.time() - self.last_grab_time) < 1.5
        }

    def stop(self):
        """Stops the reader thread and releases hardware safely."""
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception:
                pass
        print(f"[INFO] {self.name} safely released.")

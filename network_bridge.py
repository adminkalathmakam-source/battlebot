"""
Ultra-Low Latency UDP Video & Telemetry Bridge for Dual-Laptop Arena Tracking.
Streams both tracked coordinates AND compressed camera video frames between laptops
so both laptops display both live camera feeds in real time with <2ms latency.
"""

import struct
import json
import socket
import threading
import time
import cv2
import numpy as np
from config import NetworkConfig

# Packet format:
# [MAGIC 4B: 0x43, 0x42, 0x54, 0x31] ("CBT1")
# [JSON_LEN 4B (unsigned int)]
# [JPEG_LEN 4B (unsigned int)]
# [JSON_BYTES...]
# [JPEG_BYTES...]

MAGIC_HEADER = b"CBT1"
HEADER_FORMAT = "!4sII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

class UDPCornerSender:
    """Streams telemetry + compressed camera JPEG frames over UDP."""
    def __init__(self, target_ip: str, target_port: int = 5005):
        self.target_ip = target_ip
        self.target_port = target_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self.target_ip.endswith(".255"):
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send_frame_and_telemetry(self, telemetry: dict, frame_bgr: np.ndarray = None, estop: bool = False):
        """Packs telemetry JSON and compressed JPEG video frame into a single UDP datagram."""
        telemetry_payload = dict(telemetry)
        telemetry_payload["estop"] = estop
        telemetry_payload["timestamp"] = time.time()
        
        json_bytes = json.dumps(telemetry_payload).encode('utf-8')
        
        jpeg_bytes = b""
        if frame_bgr is not None:
            # Downscale frame for ultra-fast Wi-Fi transmission (~360x200 @ quality 50 is ~5KB)
            th = 200
            tw = int(th * (frame_bgr.shape[1] / frame_bgr.shape[0]))
            small = cv2.resize(frame_bgr, (tw, th))
            _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 52])
            jpeg_bytes = buf.tobytes()

        header = struct.pack(HEADER_FORMAT, MAGIC_HEADER, len(json_bytes), len(jpeg_bytes))
        datagram = header + json_bytes + jpeg_bytes
        
        try:
            self.sock.sendto(datagram, (self.target_ip, self.target_port))
            return True
        except Exception as e:
            return False

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class UDPCornerReceiver:
    """Receives and decodes telemetry + remote camera JPEG frames in background thread."""
    def __init__(self, listen_port: int = 5005):
        self.listen_port = listen_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", self.listen_port))
        self.sock.settimeout(0.5)

        self.latest_telemetry = None
        self.latest_remote_frame = None
        self.last_recv_time = 0.0
        self.sender_address = None
        self.stopped = False
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._listen_loop, daemon=True, name="UDP-Video-Receiver")
        self.thread.start()

    def _listen_loop(self):
        while not self.stopped:
            try:
                data, addr = self.sock.recvfrom(65535)
                if len(data) < HEADER_SIZE:
                    continue
                
                magic, json_len, jpeg_len = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
                if magic != MAGIC_HEADER:
                    continue
                
                json_start = HEADER_SIZE
                json_end = json_start + json_len
                jpeg_end = json_end + jpeg_len
                
                json_bytes = data[json_start:json_end]
                telemetry = json.loads(json_bytes.decode('utf-8'))
                
                remote_frame = None
                if jpeg_len > 0:
                    jpeg_bytes = data[json_end:jpeg_end]
                    np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    remote_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                with self.lock:
                    self.latest_telemetry = telemetry
                    if remote_frame is not None:
                        self.latest_remote_frame = remote_frame
                    self.last_recv_time = time.time()
                    self.sender_address = addr
            except socket.timeout:
                continue
            except Exception:
                continue

    def get_latest(self, max_staleness_sec: float = 0.45):
        """Returns (telemetry_dict, remote_frame_bgr, is_fresh_bool, sender_ip_str)."""
        with self.lock:
            if self.latest_telemetry is None:
                return None, None, False, None
            age = time.time() - self.last_recv_time
            is_fresh = (age <= max_staleness_sec)
            sender_ip = self.sender_address[0] if self.sender_address else None
            return self.latest_telemetry, self.latest_remote_frame, is_fresh, sender_ip

    def close(self):
        self.stopped = True
        try:
            self.sock.close()
        except Exception:
            pass

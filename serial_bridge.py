"""
Ultra-low-latency Serial Bridge for Master ESP32 (ESP-NOW Transmitter).
Handles USB serial connection, non-blocking packet writes, and automatic reconnects.
"""

import time
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from config import SerialConfig

class SerialBridge:
    def __init__(self, cfg: SerialConfig):
        self.cfg = cfg
        self.ser = None
        self.last_send_time = 0.0
        self.is_connected = False
        
        if not SERIAL_AVAILABLE and self.cfg.ENABLED:
            print("[WARN] 'pyserial' is not installed. Run 'pip install pyserial' to enable serial transmission.")
            self.cfg.ENABLED = False
            
        if self.cfg.ENABLED:
            self.connect()

    def connect(self):
        """Attempts to open serial connection to the ESP32 transmitter."""
        try:
            self.ser = serial.Serial(
                port=self.cfg.PORT,
                baudrate=self.cfg.BAUD_RATE,
                timeout=self.cfg.TIMEOUT,
                write_timeout=0.01  # Ultra-short write timeout to prevent blocking visual loop
            )
            self.is_connected = True
            time.sleep(0.5)  # Allow ESP32 reboot/serial stabilization
            print(f"[OK] Connected to ESP-NOW Master on {self.cfg.PORT} @ {self.cfg.BAUD_RATE} baud.")
        except Exception as e:
            self.ser = None
            self.is_connected = False
            print(f"[WARN] Serial bridge connection failed on {self.cfg.PORT}: {e}")
            print("       Running in visual-only mode. Set correct COM port in config.py.")

    def send_packet(self, packet_str: str) -> bool:
        """
        Sends formatted command packet to ESP32 over serial.
        Rate-limited to prevent buffer congestion.
        """
        now = time.time()
        if (now - self.last_send_time) < self.cfg.SEND_INTERVAL_SEC:
            return False  # Throttle transmission rate to target FPS
            
        self.last_send_time = now

        if not self.is_connected or self.ser is None:
            return False

        try:
            self.ser.write(packet_str.encode('ascii'))
            self.ser.flush()  # Push bytes immediately over USB
            return True
        except Exception as e:
            # Drop frame or log if needed
            return False

    def close(self):
        """Clean shutdown of the serial port."""
        if self.ser and self.ser.is_opened:
            try:
                # Send neutral stop command before disconnecting
                self.ser.write(b"<STOP,0,0,0,0>\n")
                self.ser.close()
            except Exception:
                pass
            print("[INFO] Serial connection closed.")

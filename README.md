# Distributed Dual-Laptop Combat Arena Tracking System

Designed for 2 corner-mounted ASUS TUF gaming laptops running OpenCV, low-latency UDP socket networking, and PySerial transmission to an ESP-NOW combat robot.

---

## 🎯 Key Capabilities
1. **No-Tag / Photo Template Tracking**:
   - Zero ArUco tags or stickers required.
   - Loads a reference top-down photo/template of your combat robot (`my_robot_template.png`) or lets you crop your robot directly from the live camera feed by pressing **`t`**.
   - Uses multi-angle rotation template matching to determine your robot's exact **$(X, Y)$ coordinates and heading orientation ($\theta$)**.
   - Segmenting non-white contrast blobs against the white floor designates any other combat robot as the **Enemy Robot**.
2. **Dual-Laptop Real-Time UDP Bridge**:
   - **Laptop 2 (Secondary Node)**: Tracks the arena from its corner, warps perspective to standardized $(800 \times 800)$ coordinates, and broadcasts telemetry packets over UDP in $<1\ \text{ms}$.
   - **Laptop 1 (Master Node)**: Tracks the arena from its opposite corner, receives Laptop 2's UDP coordinate stream, and applies **automatic occlusion fallback**.
3. **ESP-NOW Link via PySerial**:
   - Master Laptop transmits `<COMMAND,LeftPWM,RightPWM,Distance,HeadingError>\n` over USB serial to the ESP-NOW transmitter module.

---

## ⚙️ Architecture & Data Flow

```
[Corner B: Laptop 2]
    │  Webcam B (DirectShow)
    ▼
[Homography Warp: 800x800]
    │
    ▼
[Visual Photo-Template Tracker] (Finds My Bot & Enemy Bot)
    │
    ▼
[UDP Sender: Socket Port 5005] ────── (Local Wi-Fi / Hotspot) ─────┐
                                                                   │
                                                                   ▼
[Corner A: Laptop 1 (Master)] ◄─────────── [UDP Receiver: Port 5005]
    │  Webcam A (DirectShow)                                       │
    ▼                                                              │
[Homography Warp: 800x800]                                         │
    │                                                              │
    ▼                                                              │
[Visual Photo-Template Tracker]                                    │
    │                                                              │
    ▼                                                              │
[Dual-Corner Coordinate Fusion & Fallback Engine] ◄────────────────┘
    │  • Uses Cam 1 as primary
    │  • If Cam 1 loses sight/occluded -> Seamlessly falls back to Laptop 2!
    ▼
[Navigator: Vector & Steering Math]
    │  -> ΔX, ΔY, Distance, Relative Heading Error Angle
    │  -> FORWARD, TURN_LEFT, TURN_RIGHT, RAM
    ▼
[PySerial USB Bridge] -> [ESP32 Master Transmitter] -> ESP-NOW (<2ms) -> [Combat Bot]
```

---

## 🚀 Setup & Execution Guide

### 1. Network Connection (Laptop to Laptop)
Connect both ASUS TUF laptops to the **same local Wi-Fi router** or have one laptop turn on a **Windows Mobile Hotspot** and connect the other laptop to it.

Find Laptop 1's (Master) IP address:
```bash
ipconfig
# Note the IPv4 Address (e.g., 192.168.137.1 or 192.168.1.50)
```

### 2. On Master Laptop (ASUS TUF 1 - Corner A + ESP32 USB)
Run the Master Node:
```bash
cd C:\Users\user
python master_node.py
```
*(Optional flags: `--cam-idx 0` `--serial-port COM3`)*

### 3. On Secondary Laptop (ASUS TUF 2 - Corner B)
Run the Secondary Node and point it to Master Laptop's IP:
```bash
cd C:\Users\user
python secondary_node.py --master-ip 192.168.1.50
```
*(Replace `192.168.1.50` with Laptop 1's actual IP).*

---

## 🎮 Interactive Hotkeys (On Both Laptops)
- **`c`**: Launch interactive 4-corner arena calibration (Click: TL $\to$ TR $\to$ BR $\to$ BL, then press `ENTER`).
- **`t`**: Interactive Template Cropper: Drag a box around your robot on the screen to instantly set it as the reference template!
- **`q`** or **`ESC`**: Quit and release webcam cleanly.

---

## 📦 Files in this System
- [**`master_node.py`**](file:///C:/Users/user/master_node.py): Master laptop runner (Corner A + UDP Receiver + Fusion + ESP-NOW Serial).
- [**`secondary_node.py`**](file:///C:/Users/user/secondary_node.py): Secondary laptop runner (Corner B + Visual Tracker + UDP Streamer).
- [**`visual_tracker.py`**](file:///C:/Users/user/visual_tracker.py): Photo template matching with rotation search + floor contrast segmentation.
- [**`network_bridge.py`**](file:///C:/Users/user/network_bridge.py): UDP Sender & Receiver classes.
- [**`homography.py`**](file:///C:/Users/user/homography.py): 4-point perspective warp calibrator.
- [**`navigator.py`**](file:///C:/Users/user/navigator.py): Vector & differential steering calculations.
- [**`config.py`**](file:///C:/Users/user/config.py): System parameters (IP, port, serial COM, template path).
- [**`my_robot_template.png`**](file:///C:/Users/user/my_robot_template.png): Reference template photo of our robot.

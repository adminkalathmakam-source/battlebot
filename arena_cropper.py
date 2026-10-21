"""
Interactive Drag-and-Warp Arena Calibrator.
Provides a visual 4-corner draggable crop and perspective warping tool with:
- Draggable corner pins (TL, TR, BR, BL) on raw camera feed
- Real-time live top-down warped arena floor preview
- Zoom magnifier loupe (3x) for sub-pixel pin placement
- On-screen clickable buttons (Save, Reset, Full-Frame, Cancel)
- Keyboard controls (Enter/Space to save, Arrows/WASD to nudge, 1-4 to select, R to reset, Esc to cancel)
- Responsive display scaling to fit any laptop resolution
"""

import cv2
import numpy as np

class InteractiveArenaWarper:
    def __init__(self, warp_width: int = 800, warp_height: int = 800):
        self.warp_width = warp_width
        self.warp_height = warp_height
        self.dst_pts = np.array([
            [0, 0],
            [self.warp_width - 1, 0],
            [self.warp_width - 1, self.warp_height - 1],
            [0, self.warp_height - 1]
        ], dtype=np.float32)

    def calibrate(self, frame: np.ndarray, cam_source = None, existing_pts: list = None,
                  window_title: str = "Arena Perspective Crop & Warp") -> tuple:
        """
        Runs interactive 4-corner crop & warp tool.
        - frame: initial static frame (or fallback if cam_source is None)
        - cam_source: optional CameraStream or object with .read() returning (bool, frame)
        - existing_pts: list of 4 [x, y] points if previously calibrated
        - window_title: window title string
        Returns: (homography_matrix, [[x, y], ...]) or (None, None)
        """
        if frame is None and cam_source is not None:
            ret, frame = cam_source.read()
            if not ret or frame is None:
                return None, None
        elif frame is None:
            return None, None

        fh, fw = frame.shape[:2]

        # 4 Corner Points in RAW camera coordinates [x, y] (Order: TL, TR, BR, BL)
        if existing_pts and len(existing_pts) == 4:
            pts = [[float(np.clip(p[0], 0, fw - 1)), float(np.clip(p[1], 0, fh - 1))] for p in existing_pts]
        else:
            # Default to centered 80% box
            pad_x = int(fw * 0.12)
            pad_y = int(fh * 0.12)
            pts = [
                [float(pad_x), float(pad_y)],               # 0: TL
                [float(fw - pad_x), float(pad_y)],          # 1: TR
                [float(fw - pad_x), float(fh - pad_y)],     # 2: BR
                [float(pad_x), float(fh - pad_y)]           # 3: BL
            ]

        # Corner styling
        corner_names = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]
        corner_colors = [
            (0, 230, 255),    # 0: TL (Bright Gold/Yellow)
            (255, 220, 0),    # 1: TR (Cyan)
            (0, 140, 255),    # 2: BR (Bright Orange)
            (50, 255, 100)    # 3: BL (Lime Green)
        ]

        # Calculate responsive window sizing to guarantee fitting any screen
        # Max display bounds for the entire window
        header_h = 44
        footer_h = 56
        max_total_w = 1240
        max_feed_h = 540

        # Feed aspect ratio
        aspect = fw / float(fh)
        
        # We place Raw Feed on Left, Warped Square Preview on Right
        # Total width = disp_w + disp_warp_h <= max_total_w
        # disp_w = disp_feed_h * aspect
        # disp_feed_h * (aspect + 1) <= max_total_w
        disp_feed_h = int(min(max_feed_h, max_total_w / (aspect + 1.0)))
        disp_feed_w = int(disp_feed_h * aspect)
        disp_warp_size = disp_feed_h  # Square warped preview matching feed height

        total_w = disp_feed_w + disp_warp_size
        total_h = header_h + disp_feed_h + footer_h

        # Scale factor from raw frame to displayed feed
        scale_x = disp_feed_w / float(fw)
        scale_y = disp_feed_h / float(fh)

        # State tracking
        selected_idx = 0
        hover_idx = -1
        is_dragging = False
        saved = False
        status_msg = "Drag 4 corner pins to match arena walls | Use [1-4] + Arrow keys to nudge"

        # Clickable Footer Buttons: [(name, (x1, y1, x2, y2), bg_color, action_code)]
        btn_w = 175
        btn_gap = 14
        start_x = (total_w - (4 * btn_w + 3 * btn_gap)) // 2
        by1 = header_h + disp_feed_h + 10
        by2 = by1 + 36

        buttons = [
            ("SAVE & APPLY", (start_x, by1, start_x + btn_w, by2), (0, 160, 70), "save"),
            ("RESET BOX", (start_x + (btn_w + btn_gap), by1, start_x + 2 * btn_w + btn_gap, by2), (30, 110, 180), "reset"),
            ("FULL FRAME", (start_x + 2 * (btn_w + btn_gap), by1, start_x + 3 * btn_w + 2 * btn_gap, by2), (160, 120, 20), "full"),
            ("CANCEL", (start_x + 3 * (btn_w + btn_gap), by1, start_x + 4 * btn_w + 3 * btn_gap, by2), (40, 40, 160), "cancel")
        ]

        def raw_to_disp(rx, ry):
            """Convert raw frame coordinates to display screen coordinates."""
            dx = int(rx * scale_x)
            dy = int(ry * scale_y) + header_h
            return dx, dy

        def disp_to_raw(dx, dy):
            """Convert display screen coordinates to raw frame coordinates."""
            rx = float(dx) / scale_x
            ry = float(dy - header_h) / scale_y
            return np.clip(rx, 0, fw - 1), np.clip(ry, 0, fh - 1)

        def find_closest_corner_disp(dx, dy, max_dist=30):
            """Finds closest corner in display space."""
            for i, p in enumerate(pts):
                cx, cy = raw_to_disp(p[0], p[1])
                dist = np.hypot(cx - dx, cy - dy)
                if dist <= max_dist:
                    return i
            return -1

        def check_button_click(dx, dy):
            """Checks if a footer button was clicked."""
            for name, (x1, y1, x2, y2), color, action in buttons:
                if x1 <= dx <= x2 and y1 <= dy <= y2:
                    return action
            return None

        # Mouse Callback
        def on_mouse(event, x, y, flags, param):
            nonlocal selected_idx, hover_idx, is_dragging, saved, pts, status_msg

            # Check footer button hover/click
            if y > header_h + disp_feed_h:
                hover_idx = -1
                if event == cv2.EVENT_LBUTTONDOWN:
                    action = check_button_click(x, y)
                    if action == "save":
                        saved = True
                    elif action == "reset":
                        reset_corners()
                    elif action == "full":
                        full_frame_corners()
                    elif action == "cancel":
                        saved = False
                        # Trigger exit
                        pts.clear()
                return

            # Only handle raw feed area (x <= disp_feed_w and header_h <= y <= header_h + disp_feed_h)
            if header_h <= y <= header_h + disp_feed_h and x <= disp_feed_w:
                close_idx = find_closest_corner_disp(x, y, max_dist=24)
                hover_idx = close_idx

                if event == cv2.EVENT_LBUTTONDOWN:
                    if close_idx != -1:
                        selected_idx = close_idx
                        is_dragging = True
                        status_msg = f"Dragging {corner_names[selected_idx]}... Move to arena corner"
                    else:
                        # Snap closest corner to clicked point
                        min_i = 0
                        min_d = float('inf')
                        for i, p in enumerate(pts):
                            cx, cy = raw_to_disp(p[0], p[1])
                            d = np.hypot(cx - x, cy - y)
                            if d < min_d:
                                min_d = d
                                min_i = i
                        selected_idx = min_i
                        rx, ry = disp_to_raw(x, y)
                        pts[selected_idx] = [rx, ry]
                        is_dragging = True
                        status_msg = f"Snapped {corner_names[selected_idx]} to ({int(rx)}, {int(ry)})"

                elif event == cv2.EVENT_MOUSEMOVE:
                    if is_dragging and selected_idx != -1:
                        rx, ry = disp_to_raw(x, y)
                        pts[selected_idx] = [rx, ry]
                        status_msg = f"{corner_names[selected_idx]}: Raw ({int(rx)}, {int(ry)})"

                elif event == cv2.EVENT_LBUTTONUP:
                    is_dragging = False

        def reset_corners():
            nonlocal pts, status_msg
            pad_x = int(fw * 0.12)
            pad_y = int(fh * 0.12)
            pts[0] = [float(pad_x), float(pad_y)]
            pts[1] = [float(fw - pad_x), float(pad_y)]
            pts[2] = [float(fw - pad_x), float(fh - pad_y)]
            pts[3] = [float(pad_x), float(fh - pad_y)]
            status_msg = "Reset corners to centered square arena."

        def full_frame_corners():
            nonlocal pts, status_msg
            mx = int(fw * 0.02)
            my = int(fh * 0.02)
            pts[0] = [float(mx), float(my)]
            pts[1] = [float(fw - mx), float(my)]
            pts[2] = [float(fw - mx), float(fh - my)]
            pts[3] = [float(mx), float(fh - my)]
            status_msg = "Expanded corners to full camera frame."

        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_title, on_mouse)

        curr_frame = frame

        while True:
            # Check if user cancelled via button
            if len(pts) == 0:
                saved = False
                break
            if saved:
                break

            # If live camera is available, grab latest frame
            if cam_source is not None:
                ret, live_f = cam_source.read()
                if ret and live_f is not None:
                    curr_frame = live_f

            # Master composite canvas
            canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            canvas[:] = (14, 16, 22)  # Dark sleek background

            # --- 1. HEADER BAR ---
            cv2.rectangle(canvas, (0, 0), (total_w, header_h), (22, 26, 36), -1)
            cv2.putText(canvas, "ARENA PERSPECTIVE WARP & CROP TOOL", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "[1-4]: Select Corner | [Arrows / WASD]: Nudge 1px | [SPACE/ENTER]: Save | [ESC]: Cancel",
                        (390, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 200), 1, cv2.LINE_AA)

            # --- 2. RAW CAMERA FEED (LEFT PANE) ---
            feed_draw = curr_frame.copy()

            # 2a. Draw 3x3 Internal Perspective Grid
            # Interpolate perspective grid inside the 4 corners
            p0, p1, p2, p3 = [np.array(p, dtype=np.float32) for p in pts]
            for u in [0.33, 0.67]:
                top_u = (1 - u) * p0 + u * p1
                bot_u = (1 - u) * p3 + u * p2
                cv2.line(feed_draw, (int(top_u[0]), int(top_u[1])), (int(bot_u[0]), int(bot_u[1])),
                         (60, 100, 70), 1, cv2.LINE_AA)

                left_v = (1 - u) * p0 + u * p3
                right_v = (1 - u) * p1 + u * p2
                cv2.line(feed_draw, (int(left_v[0]), int(left_v[1])), (int(right_v[0]), int(right_v[1])),
                         (60, 100, 70), 1, cv2.LINE_AA)

            # 2b. Translucent fill for cropped arena
            poly_pts = np.array(pts, dtype=np.int32)
            mask_layer = feed_draw.copy()
            cv2.fillPoly(mask_layer, [poly_pts], (40, 255, 120))
            cv2.addWeighted(mask_layer, 0.16, feed_draw, 0.84, 0, feed_draw)

            # 2c. Outer Boundary Lines
            cv2.polylines(feed_draw, [poly_pts], isClosed=True, color=(0, 255, 120), thickness=2, lineType=cv2.LINE_AA)

            # 2d. Corner Pin Handles
            for i, p in enumerate(pts):
                px, py = int(p[0]), int(p[1])
                is_sel = (i == selected_idx)
                is_hov = (i == hover_idx)

                # Outer halo if selected or hovered
                radius = 16 if (is_sel or is_hov) else 12
                ring_color = (255, 255, 255) if is_sel else (0, 0, 0)
                cv2.circle(feed_draw, (px, py), radius + 4, ring_color, 2, cv2.LINE_AA)
                cv2.circle(feed_draw, (px, py), radius, corner_colors[i], -1, cv2.LINE_AA)
                cv2.circle(feed_draw, (px, py), 4, (0, 0, 0), -1, cv2.LINE_AA)

                # Label tag
                lbl = corner_names[i]
                tag_x = px + 14 if px < fw - 120 else px - 110
                tag_y = py - 12 if py > 30 else py + 24
                cv2.rectangle(feed_draw, (tag_x - 4, tag_y - 14), (tag_x + 95, tag_y + 4), (10, 12, 18), -1)
                cv2.putText(feed_draw, lbl, (tag_x, tag_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, corner_colors[i], 1, cv2.LINE_AA)

            # 2e. Magnifier Loupe on Active/Selected Corner (3x Zoom)
            if 0 <= selected_idx < 4:
                sp = pts[selected_idx]
                sx, sy = int(sp[0]), int(sp[1])
                loupe_size = 60  # 60x60 in raw pixels -> 150x150 display
                lx1 = max(0, sx - loupe_size // 2)
                ly1 = max(0, sy - loupe_size // 2)
                lx2 = min(fw, lx1 + loupe_size)
                ly2 = min(fh, ly1 + loupe_size)
                
                crop_patch = curr_frame[ly1:ly2, lx1:lx2]
                if crop_patch.shape[0] > 0 and crop_patch.shape[1] > 0:
                    loupe_disp_size = 130
                    loupe_img = cv2.resize(crop_patch, (loupe_disp_size, loupe_disp_size), interpolation=cv2.INTER_LINEAR)
                    # Loupe crosshair
                    lc = loupe_disp_size // 2
                    cv2.line(loupe_img, (lc, 0), (lc, loupe_disp_size), (0, 255, 255), 1)
                    cv2.line(loupe_img, (0, lc), (loupe_disp_size, lc), (0, 255, 255), 1)
                    cv2.rectangle(loupe_img, (0, 0), (loupe_disp_size - 1, loupe_disp_size - 1), corner_colors[selected_idx], 2)
                    
                    # Position loupe in bottom-left or top-left of feed
                    pos_x = 12
                    pos_y = fh - loupe_disp_size - 12
                    feed_draw[pos_y:pos_y + loupe_disp_size, pos_x:pos_x + loupe_disp_size] = loupe_img
                    cv2.putText(feed_draw, f"ZOOM (3X) - {corner_names[selected_idx]}",
                                (pos_x, pos_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)

            # Resize feed_draw to display dimensions
            feed_resized = cv2.resize(feed_draw, (disp_feed_w, disp_feed_h), interpolation=cv2.INTER_LINEAR)
            canvas[header_h:header_h + disp_feed_h, 0:disp_feed_w] = feed_resized

            # --- 3. LIVE WARPED TOP-DOWN PREVIEW (RIGHT PANE) ---
            src_arr = np.array(pts, dtype=np.float32)
            H, _ = cv2.findHomography(src_arr, self.dst_pts)

            if H is not None:
                warped_raw = cv2.warpPerspective(curr_frame, H, (self.warp_width, self.warp_height))
            else:
                warped_raw = np.zeros((self.warp_height, self.warp_width, 3), dtype=np.uint8)

            warped_disp = cv2.resize(warped_raw, (disp_warp_size, disp_warp_size), interpolation=cv2.INTER_LINEAR)

            # Draw top-down verification grid (4x4)
            grid_n = 4
            step = disp_warp_size // grid_n
            for g in range(1, grid_n):
                cv2.line(warped_disp, (g * step, 0), (g * step, disp_warp_size), (50, 70, 80), 1, cv2.LINE_AA)
                cv2.line(warped_disp, (0, g * step), (disp_warp_size, g * step), (50, 70, 80), 1, cv2.LINE_AA)

            # Center target crosshair
            wc = disp_warp_size // 2
            cv2.drawMarker(warped_disp, (wc, wc), (0, 255, 200), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

            # Warped Preview Header Banner
            cv2.rectangle(warped_disp, (0, 0), (disp_warp_size, 34), (16, 20, 28), -1)
            cv2.putText(warped_disp, "TOP-DOWN WARPED ARENA (FLATTENED)", (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.rectangle(warped_disp, (0, 0), (disp_warp_size - 1, disp_warp_size - 1), (0, 255, 200), 1)

            canvas[header_h:header_h + disp_warp_size, disp_feed_w:disp_feed_w + disp_warp_size] = warped_disp

            # Vertical separator line
            cv2.line(canvas, (disp_feed_w, header_h), (disp_feed_w, header_h + disp_feed_h), (40, 50, 65), 2)

            # --- 4. FOOTER ACTION BAR ---
            cv2.rectangle(canvas, (0, header_h + disp_feed_h), (total_w, total_h), (18, 20, 28), -1)

            # Status / Help text
            cv2.putText(canvas, status_msg, (16, header_h + disp_feed_h + 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 215, 230), 1, cv2.LINE_AA)

            # Render Clickable Buttons
            for name, (bx1, by1, bx2, by2), bg_col, _ in buttons:
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg_col, -1)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
                # Centered text in button
                t_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
                tx = bx1 + (btn_w - t_size[0]) // 2
                ty = by1 + (36 + t_size[1]) // 2
                cv2.putText(canvas, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # Display window
            cv2.imshow(window_title, canvas)

            # Keyboard Handling
            key_raw = cv2.waitKeyEx(25)
            if key_raw == -1:
                continue

            key = key_raw & 0xFF

            # [ENTER] (13) or [SPACE] (32) -> Save
            if key in [13, 32]:
                saved = True
                break

            # [ESC] (27) -> Cancel
            elif key == 27:
                saved = False
                break

            # [R] -> Reset
            elif key in [ord('r'), ord('R')]:
                reset_corners()

            # [F] -> Full Frame
            elif key in [ord('f'), ord('F')]:
                full_frame_corners()

            # [1, 2, 3, 4] -> Select Corner
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                selected_idx = key - ord('1')
                status_msg = f"Selected {corner_names[selected_idx]}. Drag or use arrow keys."

            # Arrow Keys / WASD Nudge
            elif 0 <= selected_idx < 4:
                nudge_step = 2.0
                dx = 0.0
                dy = 0.0

                # OpenCV waitKeyEx arrow codes on Windows:
                # Left: 2424832 (0x250000), Up: 2490368 (0x260000), Right: 2555904 (0x270000), Down: 2621440 (0x280000)
                if key_raw in [2424832, 65361] or key in [ord('a'), ord('A'), ord('h')]:
                    dx = -nudge_step
                elif key_raw in [2555904, 65363] or key in [ord('d'), ord('D'), ord('l')]:
                    dx = nudge_step
                elif key_raw in [2490368, 65362] or key in [ord('w'), ord('W'), ord('k')]:
                    dy = -nudge_step
                elif key_raw in [2621440, 65364] or key in [ord('s'), ord('S'), ord('j')]:
                    dy = nudge_step

                if dx != 0.0 or dy != 0.0:
                    pts[selected_idx][0] = float(np.clip(pts[selected_idx][0] + dx, 0, fw - 1))
                    pts[selected_idx][1] = float(np.clip(pts[selected_idx][1] + dy, 0, fh - 1))
                    status_msg = f"Nudged {corner_names[selected_idx]} -> ({int(pts[selected_idx][0])}, {int(pts[selected_idx][1])})"

        cv2.destroyWindow(window_title)
        cv2.waitKey(1)  # Flush OpenCV GUI event loop

        if saved and H is not None and len(pts) == 4:
            return H, pts
        return None, None

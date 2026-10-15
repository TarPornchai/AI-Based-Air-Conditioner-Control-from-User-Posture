import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

# ── Setup ────────────────────────────────────────────────
yolo = YOLO("yolo11n.pt")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Zone Config ──────────────────────────────────────────
NUM_ZONES = 3
TEMP_DEFAULT = 24
TEMP_MIN = 16
TEMP_MAX = 32
ADJUST_INTERVAL = 3.0  # วินาที

zone_temps = [TEMP_DEFAULT] * NUM_ZONES
zone_status = ["empty"] * NUM_ZONES   # empty / normal / hot / cold
zone_last_adjust = [time.time()] * NUM_ZONES

# ── Motor Config ─────────────────────────
SERVO_MIN = 10
SERVO_MAX = 170

current_angle = 90
target_angle = 90

SMOOTHING_ALPHA = 0.2
DEAD_ZONE = 20

sweep_direction = 1
SWEEP_SPEED = 1.5

last_seen_time = 0
TARGET_HOLD_TIME = 1.0

# สี (BGR)
COLOR_HOT    = (50,  80,  220)   # แดง
COLOR_COLD   = (220, 120,  50)   # ฟ้า
COLOR_NORMAL = (80,  180,  80)   # เขียว
COLOR_EMPTY  = (120, 120, 120)   # เทา
COLOR_BORDER_OCC = (60, 60, 220) # ขอบแดงเมื่อมีคน
ui_angle = 90
UI_SMOOTHING = 0.1  # ยิ่งน้อยยิ่งหน่วง

def status_color(s):
    return COLOR_HOT if s == "hot" else COLOR_COLD if s == "cold" else COLOR_NORMAL if s == "normal" else COLOR_EMPTY

def status_label(s):
    return "HOT" if s == "hot" else "COLD" if s == "cold" else "NORMAL" if s == "normal" else "---"

# ── Helpers ──────────────────────────────────────────────
def get_lm(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])

def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def map_x_to_angle(x, W):
    return int(np.interp(x, [0, W], [SERVO_MIN, SERVO_MAX]))

def move_servo(angle):
    angle = int(np.clip(angle, SERVO_MIN, SERVO_MAX))
    print(f"[SERVO] {angle}°", flush=True)

def update_motor(W, hot_targets):
    global current_angle, target_angle
    global sweep_direction, last_seen_time

    center = W // 2
    now = time.time()

    # มีคนร้อน
    if len(hot_targets) > 0:
        target_x = min(hot_targets, key=lambda x: abs(x - center))
        last_seen_time = now

        if abs(target_x - center) > DEAD_ZONE:
            target_angle = map_x_to_angle(target_x, W)

        current_angle = (1 - SMOOTHING_ALPHA) * current_angle + SMOOTHING_ALPHA * target_angle
        move_servo(current_angle)

    #  hold
    elif now - last_seen_time < TARGET_HOLD_TIME:
        move_servo(current_angle)

    #  sweep
    else:
        current_angle += sweep_direction * SWEEP_SPEED

        if current_angle >= SERVO_MAX:
            current_angle = SERVO_MAX
            sweep_direction = -1
        elif current_angle <= SERVO_MIN:
            current_angle = SERVO_MIN
            sweep_direction = 1

        move_servo(current_angle)

def draw_motor_ui(frame, W, H, current_angle, hot_targets):
    global ui_angle

    #  UI ค่อย ๆ ไล่ตาม angle
    ui_angle = (1 - UI_SMOOTHING) * ui_angle + UI_SMOOTHING * current_angle

    center_x = W // 2
    base_y = 50

    # ใช้ ui_angle แทน current_angle
    target_x = int(np.interp(ui_angle, [SERVO_MIN, SERVO_MAX], [0, W]))

    # วาดเส้น (บน → ล่าง)
    cv2.line(frame, (center_x, base_y), (target_x, H - 150), (255, 255, 0), 2)

    # จุดปลาย
    cv2.circle(frame, (target_x, H - 150), 8, (255, 255, 0), -1)

    #  target
    if len(hot_targets) > 0:
        for x in hot_targets:
            cv2.circle(frame, (x, H - 160), 6, (0, 0, 255), -1)

    mode = "TRACKING" if len(hot_targets) > 0 else "SWEEP"
    text = f"Motor(UI): {int(ui_angle)} deg | {mode}"

    cv2.putText(frame, text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)

# ── Classifier ───────────────────────────────────────────
def classify(landmarks, w, h):
    lm = landmarks.landmark
    l_shoulder = get_lm(lm, 11, w, h)
    r_shoulder = get_lm(lm, 12, w, h)
    l_elbow    = get_lm(lm, 13, w, h)
    r_elbow    = get_lm(lm, 14, w, h)
    l_wrist    = get_lm(lm, 15, w, h)
    r_wrist    = get_lm(lm, 16, w, h)
    l_hip      = get_lm(lm, 23, w, h)
    r_hip      = get_lm(lm, 24, w, h)
    nose       = get_lm(lm, 0,  w, h)

    shoulder_center = (l_shoulder + r_shoulder) / 2
    hip_center      = (l_hip + r_hip) / 2
    shoulder_width  = np.linalg.norm(l_shoulder - r_shoulder)
    body_height     = np.linalg.norm(shoulder_center - hip_center) + 1e-6

    cold_score = 0
    hot_score  = 0

    if shoulder_width < body_height * 0.55:
        cold_score += 2
    torso_cx = shoulder_center[0]
    wrist_mid = (abs(l_wrist[0] - torso_cx) + abs(r_wrist[0] - torso_cx)) / 2
    if wrist_mid < shoulder_width * 0.35:
        cold_score += 3
    if angle_between(l_shoulder, l_elbow, l_wrist) < 100 and \
       angle_between(r_shoulder, r_elbow, r_wrist) < 100:
        cold_score += 1

    if np.linalg.norm(l_wrist - r_wrist) > shoulder_width * 1.8:
        hot_score += 2
    for wrist in [l_wrist, r_wrist]:
        if np.linalg.norm(wrist - nose) < shoulder_width * 0.6:
            hot_score += 3
    if shoulder_width > body_height * 0.75:
        hot_score += 1

    if cold_score >= 3 and cold_score > hot_score:
        return "cold"
    elif hot_score >= 3 and hot_score > cold_score:
        return "hot"
    else:
        return "normal"

# ── Draw Zone Panel ──────────────────────────────────────
def draw_zone_panel(frame, H, W):
    panel_h = 130
    panel_y = H - panel_h
    zone_w = W // NUM_ZONES
    now = time.time()

    # พื้นหลัง panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (W, H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    for i in range(NUM_ZONES):
        x1 = i * zone_w
        x2 = x1 + zone_w - 2
        s = zone_status[i]
        t = zone_temps[i]
        col = status_color(s)
        occupied = s != "empty"

        # ขอบ zone — แดงถ้ามีคน
        border_col = COLOR_BORDER_OCC if occupied else (60, 60, 60)
        border_w = 2 if occupied else 1
        cv2.rectangle(frame, (x1 + 4, panel_y + 4), (x2 - 4, H - 4), border_col, border_w)

        # Zone label
        cv2.putText(frame, f"ZONE {i+1}",
                    (x1 + 14, panel_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # อุณหภูมิ (ใหญ่)
        temp_str = f"{t}C"
        cv2.putText(frame, temp_str,
                    (x1 + 14, panel_y + 72),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, col, 2)

        # degree symbol แยก (เล็กกว่า)
        tw, _ = cv2.getTextSize(temp_str, cv2.FONT_HERSHEY_DUPLEX, 1.6, 2)

        # status badge
        label = status_label(s)
        cv2.putText(frame, label,
                    (x1 + 14, panel_y + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        # timer progress bar
        if occupied and s != "normal":
            elapsed = now - zone_last_adjust[i]
            progress = min(1.0, elapsed / ADJUST_INTERVAL)
            bar_x1 = x1 + 14
            bar_x2 = x2 - 14
            bar_y = H - 14
            cv2.rectangle(frame, (bar_x1, bar_y - 4), (bar_x2, bar_y), (60, 60, 60), -1)
            fill_x = int(bar_x1 + (bar_x2 - bar_x1) * progress)
            cv2.rectangle(frame, (bar_x1, bar_y - 4), (fill_x, bar_y), col, -1)

        # เส้นแบ่ง zone
        if i < NUM_ZONES - 1:
            cv2.line(frame, (x2 + 1, panel_y + 8), (x2 + 1, H - 8), (60, 60, 60), 1)

    return panel_y

# ── Update Temperatures ──────────────────────────────────
def update_temps():
    now = time.time()
    for i in range(NUM_ZONES):
        s = zone_status[i]
        if s == "hot":
            if now - zone_last_adjust[i] >= ADJUST_INTERVAL:
                zone_temps[i] = max(TEMP_MIN, zone_temps[i] - 1)
                zone_last_adjust[i] = now
                print(f"Zone {i+1}: ร้อน → ลดอุณหภูมิเหลือ {zone_temps[i]}°C", flush=True)
        elif s == "cold":
            if now - zone_last_adjust[i] >= ADJUST_INTERVAL:
                zone_temps[i] = min(TEMP_MAX, zone_temps[i] + 1)
                zone_last_adjust[i] = now
                print(f"Zone {i+1}: หนาว → เพิ่มอุณหภูมิเป็น {zone_temps[i]}°C", flush=True)
            

# ── Main ─────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("กด Q เพื่อออก")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        # คำนวณพื้นที่กล้อง (เหนือ panel)
        panel_h = 130
        cam_H = H - panel_h

        # แบ่ง 3 zone แนวตั้งในพื้นที่กล้อง
        zone_w = W // NUM_ZONES
        zone_bounds = [(i * zone_w, 0, (i + 1) * zone_w, cam_H) for i in range(NUM_ZONES)]

        # reset zone status ทุก frame
        new_status = ["empty"] * NUM_ZONES
        hot_targets = []

        # ── YOLO detect ──────────────────────────────────
        results = yolo(frame, classes=[0], verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)

        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            # หา zone จาก cx ของคน
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            zone_idx = min(cx // zone_w, NUM_ZONES - 1)

            # crop + pose
            pad = 20
            x1c = max(0, x1 - pad)
            y1c = max(0, y1 - pad)
            x2c = min(W, x2 + pad)
            y2c = min(H - panel_h, y2 + pad)
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue

            ch, cw = crop.shape[:2]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb_crop)

            if pose_result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    crop,
                    pose_result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(100, 100, 255), thickness=2)
                )
                frame[y1c:y2c, x1c:x2c] = crop

                status = classify(pose_result.pose_landmarks, cw, ch)
                new_status[zone_idx] = status
                if status == "hot":
                    hot_targets.append(cx)
                print(f"P{idx+1} | Zone {zone_idx+1} | พิกัด: ({cx}, {cy}) | {status.upper()}", flush=True)
            else:
                if new_status[zone_idx] == "empty":
                    new_status[zone_idx] = "normal"

            # วาดกรอบคน
            col = status_color(new_status[zone_idx])
            cv2.rectangle(frame, (x1, y1), (x2, min(y2, cam_H)), col, 2)
            tag = f"P{idx+1} {status_label(new_status[zone_idx])}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), col, -1)
            cv2.putText(frame, tag, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # อัปเดต status และ reset timer ถ้าเปลี่ยน
        for i in range(NUM_ZONES):
            if new_status[i] != zone_status[i]:
                zone_status[i] = new_status[i]
                zone_last_adjust[i] = time.time()

        # วาดเส้นแบ่ง zone บนกล้อง
        for i in range(1, NUM_ZONES):
            lx = i * zone_w
            cv2.line(frame, (lx, 0), (lx, cam_H), (60, 60, 60), 1)
            cv2.putText(frame, f"Z{i+1}", (lx + 6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        # ── อัปเดตอุณหภูมิ ────────────────────────────
        update_temps()

        # ── วาด panel ──────────────────────────────────
        update_motor(W, hot_targets)
        draw_zone_panel(frame, H, W)
        draw_motor_ui(frame, W, H, current_angle, hot_targets)
        cv2.imshow("Thermal Behavior Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

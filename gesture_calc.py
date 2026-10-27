#!/usr/bin/env python3
import cv2, time
import numpy as np
import mediapipe as mp
from math import hypot

# ---------- MediaPipe setup ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

TIP = mp_hands.HandLandmark
WRIST = mp_hands.HandLandmark.WRIST
TIP_IDS = {
    "thumb": TIP.THUMB_TIP,
    "index": TIP.INDEX_FINGER_TIP,
    "middle": TIP.MIDDLE_FINGER_TIP,
    "ring": TIP.RING_FINGER_TIP,
    "pinky": TIP.PINKY_TIP,
}
PIP_IDS = {
    "index": mp_hands.HandLandmark.INDEX_FINGER_PIP,
    "middle": mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    "ring": mp_hands.HandLandmark.RING_FINGER_PIP,
    "pinky": mp_hands.HandLandmark.PINKY_PIP,
}
THUMB_MCP = mp_hands.HandLandmark.THUMB_MCP

# ---------- Gesture helpers ----------
def finger_states(landmarks, handed, w, h, flipped=True):
    def py(p): return p.y*h
    def px(p): return p.x*w
    pts = {n: landmarks[i] for n, i in TIP_IDS.items()}
    pips= {n: landmarks[i] for n, i in PIP_IDS.items()}
    states={}
    for f in ["index","middle","ring","pinky"]:
        states[f]= py(pts[f]) < py(pips[f]) - 6
    is_right = handed == "Right"
    thumb_vec = px(pts["thumb"]) - px(landmarks[THUMB_MCP])
    if flipped:
        states["thumb"] = (thumb_vec < -12) if is_right else (thumb_vec > 12)
    else:
        states["thumb"] = (thumb_vec > 12) if is_right else (thumb_vec < -12)
    return states

def count_up(states): return sum(1 for v in states.values() if v)

def dist_px(p1,p2,w,h):
    x1,y1=p1.x*w,p1.y*h; x2,y2=p2.x*w,p2.y*h
    return ((x2-x1)**2+(y2-y1)**2)**0.5

def classify_basic(lm, states, handed, w, h):
    """Return one of: 'OK','ThumbsUp','ThumbsDown','Count:n' (0..5)"""
    d_ok = dist_px(lm[TIP.THUMB_TIP], lm[TIP.INDEX_FINGER_TIP], w, h)
    if d_ok < 40:   # tweak if needed
        return "OK"

    only_thumb = states["thumb"] and not (states["index"] or states["middle"] or states["ring"] or states["pinky"])
    wrist = lm[WRIST]; ttip = lm[TIP.THUMB_TIP]
    if only_thumb:
        if (ttip.y*h) < (wrist.y*h) - 25: return "ThumbsUp"
        if (ttip.y*h) > (wrist.y*h) + 25: return "ThumbsDown"

    up = count_up(states)  # 0..5
    return f"Count:{up}"

# ---------- Calculator state ----------
class Calc:
    def __init__(self):
        self.expr = ""
        self.result = ""
        self.last_token = None
        self.last_commit_t = 0

    def _cooldown(self, token, now, ms=600):
        if token != self.last_token:
            self.last_token, self.last_commit_t = token, now
            return True
        if (now - self.last_commit_t) >= ms:
            self.last_commit_t = now
            return True
        return False

    def push_digit(self, d, now_ms):
        token = f"D{d}"
        if not self._cooldown(token, now_ms): return
        if self.expr and self.expr[-1].isdigit():
            self.expr += str(d)
        else:
            self.expr += str(d)

    def push_op(self, op, now_ms):
        token = f"OP{op}"
        if not self._cooldown(token, now_ms): return
        if not self.expr:
            if op == "-":
                self.expr += "-"
            return
        if self.expr[-1] in "+-*":
            self.expr = self.expr[:-1] + op
        else:
            self.expr += op

    def clear(self, now_ms):
        if not self._cooldown("CLR", now_ms): return
        self.expr = ""
        self.result = ""

    def equals(self, now_ms):
        if not self._cooldown("EQ", now_ms): return
        if not self.expr: 
            self.result = ""
            return
        safe = "".join(ch for ch in self.expr if ch.isdigit() or ch in "+-*")
        try:
            val = eval(safe)
            self.result = str(val)
        except Exception:
            self.result = "Error"

# ---------- Main app ----------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FPS,30)
    if not cap.isOpened():
        print("Camera failed. Enable permissions.")
        return

    cv2.namedWindow("Gesture Calculator", cv2.WINDOW_NORMAL)
    calc = Calc()

    prev_class = None
    stable_frames = 0
    STABLE_N = 8
    LONG_HOLD_N = 30

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        flipped=True
        while True:
            ok, frame = cap.read()
            if not ok: time.sleep(0.01); continue
            if flipped: frame = cv2.flip(frame, 1)
            h,w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            gesture_text = "Show hand to enter"
            committed = None
            now_ms = int(time.time()*1000)

            if res.multi_hand_landmarks and res.multi_handedness:
                lms = res.multi_hand_landmarks[0]
                handed = res.multi_handedness[0].classification[0].label

                mp_drawing.draw_landmarks(
                    frame, lms, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                states = finger_states(lms.landmark, handed, w, h, flipped)
                klass = classify_basic(lms.landmark, states, handed, w, h)

                if klass == prev_class:
                    stable_frames += 1
                else:
                    stable_frames = 1
                    prev_class = klass

                if stable_frames == STABLE_N:
                    if klass.startswith("Count:"):
                        d = int(klass.split(":")[1])
                        calc.push_digit(d, now_ms)
                        committed = str(d)
                    elif klass == "ThumbsUp":
                        calc.push_op("+", now_ms); committed = "+"
                    elif klass == "ThumbsDown":
                        calc.push_op("-", now_ms); committed = "-"
                    elif klass == "OK":
                        calc.push_op("*", now_ms); committed = "*"

                if klass == "Count:5" and stable_frames >= LONG_HOLD_N:
                    calc.equals(now_ms); committed = "="
                    stable_frames = STABLE_N

                if klass == "Count:0" and stable_frames >= LONG_HOLD_N:
                    calc.clear(now_ms); committed = "C"
                    stable_frames = STABLE_N

                if klass.startswith("Count:"):
                    gesture_text = f"Digit {klass.split(':')[1]}"
                elif klass == "ThumbsUp":
                    gesture_text = "Operator +"
                elif klass == "ThumbsDown":
                    gesture_text = "Operator -"
                elif klass == "OK":
                    gesture_text = "Operator *"
                else:
                    gesture_text = klass

                if committed:
                    gesture_text += f"  [✔ {committed}]"

            cv2.rectangle(frame, (0,0), (w,80), (0,0,0), -1)
            cv2.putText(frame, f"Expr: {calc.expr}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, f"Res : {calc.result}", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,255,200), 2)
            cv2.putText(frame, gesture_text, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Gesture Calculator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('m'): flipped = not flipped

    cap.release()
    cv2.destroyAllWindows()

# ---- Run ----
if __name__ == "__main__":
    main()

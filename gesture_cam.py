#!/usr/bin/env python3
import cv2, time
import numpy as np
import mediapipe as mp
from math import hypot

print("START gesture demo")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

TIP = mp_hands.HandLandmark
PIP = mp_hands.HandLandmark
WRIST = mp_hands.HandLandmark.WRIST
TIP_IDS = {"thumb": TIP.THUMB_TIP, "index": TIP.INDEX_FINGER_TIP, "middle": TIP.MIDDLE_FINGER_TIP, "ring": TIP.RING_FINGER_TIP, "pinky": TIP.PINKY_TIP}
PIP_IDS = {"index": PIP.INDEX_FINGER_PIP, "middle": PIP.MIDDLE_FINGER_PIP, "ring": PIP.RING_FINGER_PIP, "pinky": PIP.PINKY_PIP}
THUMB_MCP = mp_hands.HandLandmark.THUMB_MCP

def finger_states(landmarks, handed, w, h, flipped=True):
    def py(p): return p.y*h
    def px(p): return p.x*w
    pts = {n: landmarks[i] for n,i in TIP_IDS.items()}
    pips= {n: landmarks[i] for n,i in PIP_IDS.items()}
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

def dist(p1,p2,w,h):
    x1,y1=p1.x*w,p1.y*h; x2,y2=p2.x*w,p2.y*h
    return ((x2-x1)**2+(y2-y1)**2)**0.5

def classify(lm, states, handed, w, h):
    up = count_up(states)
    d_ok = dist(lm[TIP_IDS["thumb"]], lm[TIP_IDS["index"]], w, h)
    ok = d_ok < 40
    only_thumb = states["thumb"] and not (states["index"] or states["middle"] or states["ring"] or states["pinky"])
    thumb_tip = lm[TIP_IDS["thumb"]]; wrist = lm[WRIST]
    thumb_up = (thumb_tip.y*h) < (wrist.y*h) - 25
    thumb_down = (thumb_tip.y*h) > (wrist.y*h) + 25
    if ok: return "OK"
    if only_thumb and thumb_up: return "Thumbs Up"
    if only_thumb and thumb_down: return "Thumbs Down"
    if up==0: return "Fist"
    if up==5: return "Open Palm"
    return f"{up} finger{'s' if up!=1 else ''}"

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    cap.set(cv2.CAP_PROP_FPS,30)
    print("CAP OPENED:", cap.isOpened())
    if not cap.isOpened():
        print("Enable Camera for Terminal/VS Code in macOS settings.")
        return

    ok=False
    for _ in range(10):
        ok, _ = cap.read()
        if ok: break
        time.sleep(0.03)
    print("First frame OK:", ok)
    if not ok:
        print("Camera opened but not delivering frames.")
        return

    cv2.namedWindow("Gesture Demo", cv2.WINDOW_NORMAL)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        flipped=True
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if flipped:
                frame = cv2.flip(frame, 1)
            h,w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            label="Show your hand"
            if results.multi_hand_landmarks and results.multi_handedness:
                lms = results.multi_hand_landmarks[0]
                handed = results.multi_handedness[0].classification[0].label
                mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS,
                                          mp_styles.get_default_hand_landmarks_style(),
                                          mp_styles.get_default_hand_connections_style())
                states=finger_states(lms.landmark, handed, w, h, flipped)
                gesture=classify(lms.landmark, states, handed, w, h)
                label=f"{gesture} ({handed})"
                debug=" | ".join(f"{k}:{'↑' if v else '–'}" for k,v in states.items())
                cv2.putText(frame, debug, (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.rectangle(frame,(0,0),(w,40),(0,0,0),-1)
            cv2.putText(frame,label,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
            cv2.putText(frame,"q: quit  m: mirror",(w-240,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)

            cv2.imshow("Gesture Demo", frame)
            key=cv2.waitKey(1)&0xFF
            if key==ord('q'): break
            elif key==ord('m'): flipped=not flipped

    cap.release(); cv2.destroyAllWindows()
    print("DONE")

if __name__ == "__main__":
    main()

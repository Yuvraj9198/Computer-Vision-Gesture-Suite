import cv2
import numpy as np
from ultralytics import YOLO
import math

# =============================
# MODEL
# =============================
model = YOLO("yolov8n.pt")

# =============================
# PARAMETERS (STRICT)
# =============================
MARGIN = 20
MIN_FRAMES_IN_MARGIN = 6
MIN_MOVE_PIXELS = 25
CONFIDENCE = 0.6
MIN_HEIGHT = 130  # ignore bags / partials

# =============================
# GLOBALS
# =============================
points = []
IN_COUNT = 0
OUT_COUNT = 0
tracks = {}

# =============================
# MOUSE
# =============================
def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        print("Selected:", (x, y))

# =============================
# GEOMETRY
# =============================
def create_margins(p1, p2, d):
    p1, p2 = np.array(p1), np.array(p2)
    v = p2 - p1
    n = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    return (
        (tuple((p1+n*d).astype(int)), tuple((p2+n*d).astype(int))),
        (tuple((p1-n*d).astype(int)), tuple((p2-n*d).astype(int)))
    )

def side(pt, a, b):
    return np.sign((b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0]))

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# =============================
# VIDEO
# =============================
cap = cv2.VideoCapture(0)
cv2.namedWindow("Classroom Counter (STRICT)")
cv2.setMouseCallback("Classroom Counter (STRICT)", mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=CONFIDENCE,
        iou=0.5
    )

    if len(points) == 2:
        main = points
        up, down = create_margins(points[0], points[1], MARGIN)

        cv2.line(frame, *main, (0,255,0), 2)
        cv2.line(frame, *up, (255,255,0), 2)
        cv2.line(frame, *down, (255,255,0), 2)

    if results[0].boxes.id is not None:
        for box, tid in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1,y1,x2,y2 = map(int, box)
            h = y2 - y1
            if h < MIN_HEIGHT:
                continue

            cx, cy = (x1+x2)//2, (y1+y2)//2
            tid = int(tid)

            if tid not in tracks:
                tracks[tid] = {
                    "state": "outside",
                    "frames": 0,
                    "start": (cx, cy),
                    "last": (cx, cy),
                    "counted": False
                }

            t = tracks[tid]
            t["last"] = (cx, cy)

            if len(points) == 2 and not t["counted"]:
                s_up = side((cx,cy), *up)
                s_dn = side((cx,cy), *down)
                s_main = side((cx,cy), *main)

                # ENTER MARGIN
                if t["state"] == "outside" and s_up != s_dn:
                    t["state"] = "margin"
                    t["frames"] = 1
                    t["start"] = (cx, cy)

                # STAY IN MARGIN
                elif t["state"] == "margin":
                    t["frames"] += 1

                    if s_up == s_dn:
                        t["state"] = "outside"  # left margin without crossing

                    elif t["frames"] >= MIN_FRAMES_IN_MARGIN:
                        movement = dist(t["start"], (cx,cy))

                        if movement >= MIN_MOVE_PIXELS:
                            if s_main > 0:
                                IN_COUNT += 1
                                t["counted"] = True
                            elif s_main < 0:
                                OUT_COUNT += 1
                                t["counted"] = True

                            t["state"] = "done"

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.circle(frame,(cx,cy),4,(0,255,255),-1)

    cv2.putText(frame,f"IN: {IN_COUNT}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.putText(frame,f"OUT: {OUT_COUNT}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("Classroom Counter (STRICT)", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

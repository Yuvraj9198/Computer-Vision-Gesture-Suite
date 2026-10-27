#!/usr/bin/env python3
import cv2, time, json, os
import numpy as np
import mediapipe as mp

# ==== config ====
OUT_DIR = "sign_dataset"     # created if missing
LABELS = ["hello","thank_you","yes","no","please","good","morning","night","help","love"]
FRAMES_PER_CLIP = 50         # ~2s @ ~25 fps
PAUSE_BETWEEN = 1.0          # seconds between clips

# Use MediaPipe Holistic to get BOTH hands; (21*3)*2 = 126 dims
mp_hol = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

def extract_vec(res, w, h):
    """Return 126-d vector [left(21*3), right(21*3)] (x,y,z)—zeros if hand missing."""
    def hand_vec(landmarks):
        if not landmarks:
            return np.zeros(21*3, np.float32)
        v = []
        for lm in landmarks.landmark:
            v += [lm.x, lm.y, lm.z]
        return np.array(v, np.float32)
    left = hand_vec(res.left_hand_landmarks)
    right= hand_vec(res.right_hand_landmarks)
    return np.concatenate([left, right]).astype(np.float32)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(LABELS, f, indent=2)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Camera failed. Enable permissions.")
        return
    cv2.namedWindow("Record Signs", cv2.WINDOW_NORMAL)

    with mp_hol.Holistic(static_image_mode=False, model_complexity=1,
                         enable_segmentation=False,
                         refine_face_landmarks=False) as hol:
        for label in LABELS:
            idx = 0
            while True:
                # Wait for you to press SPACE to record one clip for this label
                print(f"[{label}] Press SPACE to record a {FRAMES_PER_CLIP}-frame clip, or 'n' to move to next label.")
                while True:
                    ok, frame = cap.read()
                    if not ok: continue
                    vis = cv2.flip(frame, 1)
                    cv2.putText(vis, f"Label: {label} | Clips saved: {idx}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.putText(vis, "SPACE = record clip | n = next label | q = quit",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
                    cv2.imshow("Record Signs", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '): break
                    if key == ord('n'): break
                    if key == ord('q'): cap.release(); cv2.destroyAllWindows(); return
                if key == ord('n'): break

                # Record one clip
                seq = []
                for _ in range(FRAMES_PER_CLIP):
                    ok, frame = cap.read()
                    if not ok: continue
                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]
                    res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    vec = extract_vec(res, w, h)   # (126,)
                    seq.append(vec)
                    # optional: show basic overlay
                    cv2.putText(frame, f"Recording: {label} ({len(seq)}/{FRAMES_PER_CLIP})",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow("Record Signs", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release(); cv2.destroyAllWindows(); return

                seq = np.stack(seq, axis=0)  # (T,126)
                save_path = os.path.join(OUT_DIR, f"{label}_{idx:03d}.npy")
                np.save(save_path, seq)
                print(f"[+] saved {save_path}")
                idx += 1
                time.sleep(PAUSE_BETWEEN)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

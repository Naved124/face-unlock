import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis

EYE_A = list(range(33, 43))
EYE_B = list(range(87, 97))
MOUTH = list(range(52, 72))

CLOSED_T = 0.09            # relative: below this = closed
OPEN_T = 0.11              # relative: above this = open again
MIN_BLINK_SECONDS = 0.10   # shorter = single glitched frame, ignore
MAX_BLINK_SECONDS = 0.6    # longer = looking down, ignore
MAX_SCALE_CHANGE = 0.10    # eyes-to-mouth distance may change at most 8%
MAX_MOVE = 0.30            # face may move at most 15% of eyes-to-mouth distance

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "landmark_2d_106"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(320, 320))


def eye_height(points):
    ys = points[:, 1]
    return ys.max() - ys.min()


def measure(lm):
    """Return (relative openness, eyes-to-mouth distance, face center)."""
    eye_center = lm[EYE_A + EYE_B].mean(axis=0)
    mouth_center = lm[MOUTH].mean(axis=0)
    ref = float(np.linalg.norm(mouth_center - eye_center))
    center = (eye_center + mouth_center) / 2
    if ref <= 0:
        return 0.0, 0.0, center
    h = (eye_height(lm[EYE_A]) + eye_height(lm[EYE_B])) / 2
    return h / ref, ref, center


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

eyes_closed = False
closed_at = 0.0
blinks = 0

# face geometry from the last frame where eyes were open
last_open_ref = None
last_open_center = None
# worst-case motion seen during the current closed period
max_scale_change = 0.0
max_move = 0.0

prev_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    faces = app.get(frame)

    if len(faces) == 1:
        lm = faces[0].landmark_2d_106
        rel, ref, center = measure(lm)

        for i in EYE_A + EYE_B:
            x, y = lm[i].astype(int)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        for i in MOUTH:
            x, y = lm[i].astype(int)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        now_t = time.time()

        if not eyes_closed:
            if rel < CLOSED_T and last_open_ref:
                # eyes just closed: start watching for motion
                eyes_closed = True
                closed_at = now_t
                max_scale_change = 0.0
                max_move = 0.0
            else:
                # eyes open: remember the steady face geometry
                last_open_ref = ref
                last_open_center = center

        if eyes_closed:
            # how much did the whole face change vs. just before the eyes closed?
            scale_change = abs(ref - last_open_ref) / last_open_ref
            move = float(np.linalg.norm(center - last_open_center)) / last_open_ref
            max_scale_change = max(max_scale_change, scale_change)
            max_move = max(max_move, move)

            if rel > OPEN_T:
                eyes_closed = False
                duration = now_t - closed_at
                info = (f"{duration*1000:.0f} ms  scale {max_scale_change:.2f}"
                        f"  move {max_move:.2f}")

                if duration < MIN_BLINK_SECONDS:
                    print(f"rejected: too short     ({info})")
                elif duration > MAX_BLINK_SECONDS:
                    print(f"rejected: too long      ({info})")
                elif max_scale_change > MAX_SCALE_CHANGE or max_move > MAX_MOVE:
                    print(f"rejected: face moved    ({info})")
                else:
                    blinks += 1
                    print(f"BLINK #{blinks}            ({info})")

        state = "CLOSED" if eyes_closed else "open"
        cv2.putText(frame, f"relative {rel:.3f}  [{state}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    else:
        eyes_closed = False   # lost the face: reset
        last_open_ref = None

    now = time.time()
    fps = 1.0 / (now - prev_time)
    prev_time = now

    cv2.putText(frame, f"blinks: {blinks}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"fps: {fps:.1f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("blink", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
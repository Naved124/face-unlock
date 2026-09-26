import cv2
import numpy as np
from insightface.app import FaceAnalysis

EYE_A = list(range(33, 43))   # one eye
EYE_B = list(range(87, 97))   # the other eye

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "landmark_2d_106"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))


def openness(points):
    xs, ys = points[:, 0], points[:, 1]
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    return height / width if width > 0 else 0.0


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    faces = app.get(frame)

    if len(faces) == 1:
        lm = faces[0].landmark_2d_106   # shape (106, 2)

        # all 106 points in grey
        for x, y in lm.astype(int):
            cv2.circle(frame, (x, y), 1, (160, 160, 160), -1)

        # eye points in yellow
        for i in EYE_A + EYE_B:
            x, y = lm[i].astype(int)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

        ratio = (openness(lm[EYE_A]) + openness(lm[EYE_B])) / 2
        cv2.putText(frame, f"eye openness: {ratio:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("eyes", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
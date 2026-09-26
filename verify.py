import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

TEMPLATE_FILE = "enroll/naved.npy"
THRESHOLD = 0.5

if not os.path.exists(TEMPLATE_FILE):
    raise SystemExit(f"No template at {TEMPLATE_FILE}, run enroll.py first")

template = np.load(TEMPLATE_FILE)

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))

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
        face = faces[0]
        sim = float(np.dot(template, face.normed_embedding))
        match = sim >= THRESHOLD

        color = (0, 255, 0) if match else (0, 0, 255)
        label = f"{'MATCH' if match else 'NO MATCH'}  {sim:.2f}"

        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        cv2.putText(frame, f"DENY: need exactly 1 face (seeing {len(faces)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("verify", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
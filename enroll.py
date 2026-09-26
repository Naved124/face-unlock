import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

NUM_SAMPLES = 5
OUT_DIR = "enroll"
OUT_FILE = os.path.join(OUT_DIR, "naved.npy")

# Detection + recognition this time (recognition gives us the embedding)
app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

samples = []

while len(samples) < NUM_SAMPLES:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if len(faces) == 1:
        status = f"{len(samples)}/{NUM_SAMPLES}  press c to capture"
        color = (0, 255, 0)
    else:
        status = f"need exactly 1 face (seeing {len(faces)})"
        color = (0, 0, 255)

    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("enroll", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c") and len(faces) == 1:
        samples.append(faces[0].normed_embedding)
        print(f"captured sample {len(samples)}")

cap.release()
cv2.destroyAllWindows()

if len(samples) < NUM_SAMPLES:
    raise SystemExit("Enrollment cancelled, nothing saved")

# Average the samples, then scale back to length 1
template = np.mean(samples, axis=0)
template /= np.linalg.norm(template)

os.makedirs(OUT_DIR, exist_ok=True)
print(np.dot(samples[0], samples[1]))
np.save(OUT_FILE, template)
print(f"saved {OUT_FILE}  shape={template.shape}")
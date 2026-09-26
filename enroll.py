import cv2
import numpy as np
from insightface.app import FaceAnalysis

from identity import TEMPLATE_FILE, load_recognizer, save_template

NUM_SAMPLES = 5

# Detection only here; embeddings come from the same model challenge.py uses
app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))
rec, model_sha = load_recognizer()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

samples = []

while len(samples) < NUM_SAMPLES:
    ok, frame = cap.read()
    if not ok:
        break
    # Embed the raw camera frame (same as challenge.py); mirror only for display.
    faces = app.get(frame)

    # Draw on a copy: boxes on the raw frame would end up inside the embedded crop.
    view = frame.copy()
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
    view = cv2.flip(view, 1)

    if len(faces) == 1:
        status = f"{len(samples)}/{NUM_SAMPLES}  press c to capture"
        color = (0, 255, 0)
    else:
        status = f"need exactly 1 face (seeing {len(faces)})"
        color = (0, 0, 255)

    cv2.putText(view, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("enroll", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c") and len(faces) == 1:
        rec.get(frame, faces[0])
        samples.append(faces[0].normed_embedding)
        print(f"captured sample {len(samples)}")

cap.release()
cv2.destroyAllWindows()

if len(samples) < NUM_SAMPLES:
    raise SystemExit("Enrollment cancelled, nothing saved")

# How consistent were the samples? (all pairs; low values = a bad capture)
sims = [float(np.dot(a, b)) for i, a in enumerate(samples) for b in samples[i + 1:]]
print(f"sample agreement: min {min(sims):.2f}  mean {np.mean(sims):.2f}")

save_template(samples, model_sha)
print(f"saved {TEMPLATE_FILE}  (model {model_sha[:12]}…)")
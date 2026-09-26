import cv2
from insightface.app import FaceAnalysis

# Load only the detector for now; recognition comes in piece 3
app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 means CPU

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)

    faces = app.get(frame)          # list of Face objects (may be empty)

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)   # OpenCV wants ints
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{face.det_score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("face-unlock", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
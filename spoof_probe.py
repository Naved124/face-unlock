import sys

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

A_MODEL = "models/minifasnet_v2.onnx"   # upstream MiniFASNetV2
B_MODEL = "models/facenox_best.onnx"    # facenox MiniFAS-V2-SE

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(320, 320))

sess_a = ort.InferenceSession(A_MODEL, providers=["CPUExecutionProvider"])
sess_b = ort.InferenceSession(B_MODEL, providers=["CPUExecutionProvider"])
in_a = sess_a.get_inputs()[0].name
in_b = sess_b.get_inputs()[0].name
print("B input:", sess_b.get_inputs()[0].shape,
      " output:", sess_b.get_outputs()[0].shape)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def crop_a(frame, bbox, scale=2.7, size=80):
    """Upstream: grow box around center, shift back inside the image."""
    img_h, img_w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    scale = min((img_h - 1) / h, (img_w - 1) / w, scale)
    cx, cy = x1 + w / 2, y1 + h / 2
    left, top = cx - w * scale / 2, cy - h * scale / 2
    right, bottom = cx + w * scale / 2, cy + h * scale / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img_w - 1:
        left -= right - img_w + 1
        right = img_w - 1
    if bottom > img_h - 1:
        top -= bottom - img_h + 1
        bottom = img_h - 1
    l, t, r, b = int(left), int(top), int(right), int(bottom)
    return cv2.resize(frame[t:b + 1, l:r + 1], (size, size))


def crop_b(frame, bbox, expand=1.5, size=128):
    """facenox: square crop of the longer side, mirror-pad past the edges."""
    x1, y1, x2, y2 = bbox
    side = int(max(x2 - x1, y2 - y1) * expand)
    x = int((x1 + x2) / 2 - side / 2)
    y = int((y1 + y2) / 2 - side / 2)
    padded = cv2.copyMakeBorder(frame, side, side, side, side,
                                cv2.BORDER_REFLECT_101)
    crop = padded[y + side:y + 2 * side, x + side:x + 2 * side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def live_a(frame, bbox):
    crop = crop_a(frame, bbox)
    x = crop.astype(np.float32).transpose(2, 0, 1)[None]          # BGR, 0-255
    return softmax(sess_a.run(None, {in_a: x})[0][0])[1], crop    # index 1 = real


def live_b(frame, bbox):
    crop = crop_b(frame, bbox)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # RGB, 0-1
    return softmax(sess_b.run(None, {in_b: x})[0][0])[0], crop     # index 0 = real


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

label = " ".join(sys.argv[1:]) or "unlabelled"
scores_a, scores_b = [], []


def summary():
    """Worst and best score per model over every frame, not just printed samples."""
    if not scores_a:
        print(f"RUN SUMMARY [{label}]  no face frames")
        return
    for name, s in (("A", scores_a), ("B", scores_b)):
        s = np.array(s)
        print(f"RUN SUMMARY [{label}]  {name}  frames {len(s)}"
              f"  min {s.min():.2f}  p5 {np.percentile(s, 5):.2f}"
              f"  mean {s.mean():.2f}  max {s.max():.2f}")


frame_no = 0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = app.get(frame)
        if len(faces) == 1:
            bbox = faces[0].bbox
            a, ca = live_a(frame, bbox)
            b, cb = live_b(frame, bbox)
            scores_a.append(a)
            scores_b.append(b)

            cv2.putText(frame, f"A minifas  live {a:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"B facenox  live {b:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            view = np.hstack([cv2.resize(ca, (240, 240)), cv2.resize(cb, (240, 240))])
            cv2.imshow("model inputs: A | B", view)

            frame_no += 1
            if frame_no % 15 == 0:
                print(f"A {a:.2f}   B {b:.2f}")

        cv2.imshow("spoof probe", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    pass  # Ctrl+C still prints the summary
finally:
    cap.release()
    cv2.destroyAllWindows()
    summary()

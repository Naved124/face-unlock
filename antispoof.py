"""Passive anti-spoof check: is this face real skin, or a screen/print?

Model: facenox MiniFAS-V2-SE (quantized ONNX), Apache-2.0.
https://github.com/facenox/face-antispoof-onnx
Preprocessing mirrors their src/inference/preprocess.py exactly:
square crop of the longer bbox side x1.5, mirror-padded, 128x128, RGB, /255.
Output: 2 logits, index 0 = real.
"""
import hashlib

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = "models/facenox_best.onnx"
# Pinned hash. The service refuses to start if the file differs:
# a swapped model that always says "real" would be a silent backdoor.
MODEL_SHA256 = "fde20585635cae62ed1d41796f76b6f8bc4b92cd91ec1cf0f1bc6485d2d587a9"

EXPAND = 1.5
SIZE = 128


class ModelIntegrityError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def crop_face(frame, bbox):
    """Square crop around the face, mirror-padding wherever it runs off the frame."""
    x1, y1, x2, y2 = bbox
    side = int(max(x2 - x1, y2 - y1) * EXPAND)
    if side <= 0:
        raise ValueError("empty face box")
    x = int((x1 + x2) / 2 - side / 2)
    y = int((y1 + y2) / 2 - side / 2)
    padded = cv2.copyMakeBorder(frame, side, side, side, side, cv2.BORDER_REFLECT_101)
    crop = padded[y + side:y + 2 * side, x + side:x + 2 * side]
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


class AntiSpoof:
    def __init__(self, path=MODEL_PATH, expected_sha256=MODEL_SHA256):
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise ModelIntegrityError(
                f"{path}: sha256 {actual} does not match pinned {expected_sha256}")
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def score(self, frame, bbox):
        """Probability (0-1) that the face is real. `frame` must be un-mirrored BGR."""
        rgb = cv2.cvtColor(crop_face(frame, bbox), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        logits = self.sess.run(None, {self.input: x})[0][0]
        e = np.exp(logits - logits.max())
        return float(e[0] / e.sum())

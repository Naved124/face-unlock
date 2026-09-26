"""Identity check: does this face match the enrolled template?

Recognition model: MobileFaceNet (w600k_mbf) from InsightFace's buffalo_sc pack.
Same training data and 512-d embeddings as buffalo_l's ResNet-50, ~10x less compute.
"""
import hashlib
import os

import numpy as np
from insightface.app import FaceAnalysis

TEMPLATE_FILE = "enroll/naved.npz"
MODEL_PACK = "buffalo_sc"
MODEL_FILE = os.path.expanduser(f"~/.insightface/models/{MODEL_PACK}/w600k_mbf.onnx")
EMBED_DIM = 512


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_recognizer():
    """Return (recognition model, sha256 of its file). Downloads buffalo_sc on first use."""
    pack = FaceAnalysis(  # FaceAnalysis insists on a detector; det_500m is tiny
        name=MODEL_PACK,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    pack.prepare(ctx_id=-1, det_size=(320, 320))
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"expected recognition model at {MODEL_FILE}")
    return pack.models["recognition"], sha256_file(MODEL_FILE)


def save_template(embeddings, model_sha256, path=TEMPLATE_FILE):
    """Average normalized embeddings into one unit vector; record which model made it."""
    template = np.mean(embeddings, axis=0)
    template /= np.linalg.norm(template)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, template=template.astype(np.float32), model_sha256=np.array(model_sha256))


class Identity:
    def __init__(self, path=TEMPLATE_FILE):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No template at {path}, run enroll.py first")
        self.rec, model_sha = load_recognizer()
        with np.load(path) as data:  # allow_pickle stays False: plain arrays only
            t = data["template"]
            enrolled_sha = str(data["model_sha256"])
        # Embeddings from different models aren't comparable, and a changed model
        # file could be a swapped one: either way, refuse instead of guessing.
        if enrolled_sha != model_sha:
            raise ValueError(f"{path} was enrolled with a different recognition model "
                             f"({enrolled_sha[:12]}… vs {model_sha[:12]}…); re-enroll")
        if t.shape != (EMBED_DIM,):
            raise ValueError(f"{path}: expected shape ({EMBED_DIM},), got {t.shape}")
        norm = float(np.linalg.norm(t))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"{path}: template is empty or corrupt")
        self.template = (t / norm).astype(np.float32)

    def embed(self, frame, face):
        """Compute this face's embedding (needs face.kps from the detector)."""
        self.rec.get(frame, face)

    def similarity(self, face):
        """Cosine similarity (-1..1) between an embedded face and the template."""
        return float(np.dot(self.template, face.normed_embedding))

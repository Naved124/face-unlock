import secrets
import statistics
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from antispoof import AntiSpoof
from identity import Identity
from liveness import BlinkDetector, MAX_BLINK_SECONDS

ROUNDS = 3
DELAY_RANGE = (1.5, 4.0)  # random wait before each prompt (s)
QUIET = 1.0               # no blink allowed this long before a prompt
MIN_REACTION = 0.15       # faster than this can't be a reaction
WINDOW = 1.0              # blink must start within this after the prompt
TOTAL_TIMEOUT = 25.0      # whole challenge must finish in this time
LIVE_T = 0.5              # median anti-spoof score needed (real ~0.9+, phones <=0.02)
MATCH_T = 0.5             # median identity similarity needed (same as verify.py)
SWAP_T = 0.25             # any checked frame below this = a different person: fail now
ID_EVERY = 5              # run face recognition every Nth frame (it's the slow model)...
JUMP = 0.5                # ...plus any frame where the face jumped > 0.5 face-widths
WARMUP_FRAMES = 10        # frames run through every model before the clock starts
FILL_LIGHT = False        # white screen as a front light: tested, didn't rescue low light
                          # (anti-spoof still ~0.04), so off by default
CANVAS = (720, 1280)      # fill-light canvas size (h, w); shown fullscreen

rng = secrets.SystemRandom()

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "landmark_2d_106"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(320, 320))

spoof = AntiSpoof()   # raises if the model file doesn't match its pinned hash
ident = Identity()    # raises if the template is missing, corrupt or from another model

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

if FILL_LIGHT:
    cv2.namedWindow("challenge", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("challenge", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def show(frame, text, color, info):
    """Draw the UI. With FILL_LIGHT, a small preview sits on a white canvas."""
    view = cv2.flip(frame, 1)
    if FILL_LIGHT:
        h, w = CANVAS
        canvas = np.full((h, w, 3), 255, np.uint8)
        pw, ph = 320, 240
        x = (w - pw) // 2
        canvas[20:20 + ph, x:x + pw] = cv2.resize(view, (pw, ph))
        dark = tuple(int(c * 0.7) for c in color)
        cv2.putText(canvas, text, (x, 20 + ph + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, dark, 2)
        cv2.putText(canvas, info, (x, 20 + ph + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 80, 80), 1)
        view = canvas
    else:
        cv2.putText(view, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(view, info, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    cv2.imshow("challenge", view)
    return cv2.waitKey(1) & 0xFF


# Warm-up: onnxruntime (and the CPU clock) take several runs to reach full speed.
# Also gives the fill light a moment to light the scene and the camera to re-expose.
for _ in range(WARMUP_FRAMES):
    ok, frame = cap.read()
    if not ok:
        break
    for f in app.get(frame):
        spoof.score(frame, f.bbox)
        ident.embed(frame, f)
    show(frame, "Get ready...", (200, 200, 200), "")

detector = BlinkDetector()
start = time.time()
t_read = t_models = t_spoof = t_rec = 0.0  # seconds per stage, to find the bottleneck
state = "wait"
round_no = 1
prompt_due = start + rng.uniform(*DELAY_RANGE)
prompt_at = 0.0
last_blink = 0.0
result = ""
live = sim = 0.0
frames = id_checks = 0
force_id = True       # first frame always gets an identity check
prev_center = None
# anti-spoof: every frame. identity: every checked frame. round_*: current prompt only.
all_live, all_sim = [], []
round_live, round_sim = [], []
all_bright = []  # mean face brightness (0-255). Diagnostic only: auto-exposure
                 # normalizes it, so it can't detect a dark room (dark runs read ~115)


def stats(xs):
    if not xs:
        return "no data"
    return f"median {statistics.median(xs):.2f}  min {min(xs):.2f}  max {max(xs):.2f}"


def finish(text):
    global state, result
    state, result = "done", text
    print(text)
    if frames:  # always show what the models thought, even on timing failures
        fps = frames / (time.time() - start)
        print(f"  anti-spoof  {stats(all_live)}")
        print(f"  identity    {stats(all_sim)}   ({id_checks} checks)")
        print(f"  face brightness  {stats(all_bright)}")
        print(f"  {frames} frames, {fps:.1f} fps   per frame: camera"
              f" {t_read/frames*1000:.0f} ms, detect+landmarks {t_models/frames*1000:.0f} ms,"
              f" anti-spoof {t_spoof/frames*1000:.0f} ms;"
              f" recognition {t_rec/max(id_checks, 1)*1000:.0f} ms per check")


while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break
    now = time.time()

    # Models see the raw camera frame; the mirror flip is for display only.
    if state != "done":
        t_read += now - t0
        t0 = time.time()
        faces = app.get(frame)  # detection + landmarks only
        t_models += time.time() - t0
        frames += 1
        if now - start > TOTAL_TIMEOUT:
            finish("FAIL: timed out")
        elif len(faces) != 1:
            detector.reset()
            prev_center = None
            force_id = True  # whoever reappears gets checked immediately
            if state == "prompt":
                finish("FAIL: face lost")
        else:
            face = faces[0]
            bx1, by1, bx2, by2 = np.clip(face.bbox, 0, None).astype(int)
            patch = frame[by1:by2, bx1:bx2]
            if patch.size:
                all_bright.append(float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean()))
            t0 = time.time()
            live = spoof.score(frame, face.bbox)
            t_spoof += time.time() - t0
            all_live.append(live)
            if state == "prompt":
                round_live.append(live)

            event = detector.update(face.landmark_2d_106, now)
            if event:
                last_blink = now

            # Identity: periodic, plus every frame where a swap could hide.
            x1, y1, x2, y2 = face.bbox
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            jumped = (prev_center is not None and
                      np.linalg.norm(center - prev_center) > JUMP * (x2 - x1))
            prev_center = center
            if force_id or jumped or event or frames % ID_EVERY == 0:
                t0 = time.time()
                ident.embed(frame, face)
                t_rec += time.time() - t0
                id_checks += 1
                force_id = False
                sim = ident.similarity(face)
                all_sim.append(sim)
                if state == "prompt":
                    round_sim.append(sim)
                if sim < SWAP_T:
                    finish(f"FAIL: not the enrolled face (similarity {sim:.2f})")

            if state == "wait":
                quiet = not detector.closed and now - last_blink >= QUIET
                if now >= prompt_due and quiet:
                    state = "prompt"
                    prompt_at = now
                    round_live, round_sim = [live], []
                    force_id = True  # check who's there as the prompt starts
                    print(f"round {round_no}: BLINK NOW")

            elif state == "prompt":
                if event and event.started >= prompt_at:
                    # the blink frame always gets an identity check, so round_sim has data
                    reaction = event.started - prompt_at
                    r_live = statistics.median(round_live)
                    r_sim = statistics.median(round_sim) if round_sim else -1.0
                    print(f"  reaction {reaction*1000:.0f} ms"
                          f"  blink {event.duration*1000:.0f} ms"
                          f"  scale {event.scale:.2f}  move {event.move:.2f}"
                          f"  live {r_live:.2f}  id {r_sim:.2f}"
                          f"  [{event.reason}]")
                    if not event.ok:
                        finish(f"FAIL: bad blink ({event.reason})")
                    elif reaction < MIN_REACTION:
                        finish("FAIL: too fast")
                    elif reaction > WINDOW:
                        finish("FAIL: too late")
                    elif r_live < LIVE_T:
                        finish(f"FAIL: spoof suspected (live {r_live:.2f})")
                    elif r_sim < MATCH_T:
                        finish(f"FAIL: not the enrolled face (id {r_sim:.2f})")
                    elif round_no == ROUNDS:
                        o_live = statistics.median(all_live)
                        o_sim = statistics.median(all_sim)
                        if o_live < LIVE_T:
                            finish(f"FAIL: spoof suspected (overall {o_live:.2f})")
                        elif o_sim < MATCH_T:
                            finish(f"FAIL: not the enrolled face (overall {o_sim:.2f})")
                        else:
                            finish("PASS: live, enrolled user")
                    else:
                        round_no += 1
                        state = "wait"
                        prompt_due = now + rng.uniform(*DELAY_RANGE)
                elif now - prompt_at > WINDOW + MAX_BLINK_SECONDS:
                    finish("FAIL: no blink")

    if state == "wait":
        text, color = f"Round {round_no}/{ROUNDS}: look here", (200, 200, 200)
    elif state == "prompt":
        text, color = "BLINK NOW", (0, 0, 255)
    else:
        text = result
        color = (0, 255, 0) if result.startswith("PASS") else (0, 0, 255)

    info = f"relative {detector.rel:.3f}   live {live:.2f}   id {sim:.2f}"
    if show(frame, text, color, info) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

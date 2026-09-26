import secrets
import statistics
import time

import cv2
from insightface.app import FaceAnalysis

from antispoof import AntiSpoof
from liveness import BlinkDetector, MAX_BLINK_SECONDS

ROUNDS = 3
DELAY_RANGE = (1.5, 4.0)  # random wait before each prompt (s)
QUIET = 1.0               # no blink allowed this long before a prompt
MIN_REACTION = 0.15       # faster than this can't be a reaction
WINDOW = 1.0              # blink must start within this after the prompt
TOTAL_TIMEOUT = 25.0      # whole challenge must finish in this time
LIVE_T = 0.5              # median anti-spoof score needed (provisional: tune
                          # after backlit + lamp runs; real ~0.9+, phones <=0.02)

rng = secrets.SystemRandom()

app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "landmark_2d_106"],
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=-1, det_size=(320, 320))

spoof = AntiSpoof()  # raises if the model file doesn't match its pinned hash

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

detector = BlinkDetector()
start = time.time()
state = "wait"
round_no = 1
prompt_due = start + rng.uniform(*DELAY_RANGE)
prompt_at = 0.0
last_blink = 0.0
result = ""
live = 0.0
all_scores = []    # anti-spoof score of every frame in the challenge
round_scores = []  # frames from the current prompt until the blink ends


def finish(text):
    global state, result
    state, result = "done", text
    print(text)


while True:
    ok, frame = cap.read()
    if not ok:
        break
    now = time.time()

    # Models see the raw camera frame; the mirror flip is for display only.
    if state != "done":
        faces = app.get(frame)
        if now - start > TOTAL_TIMEOUT:
            finish("FAIL: timed out")
        elif len(faces) != 1:
            detector.reset()
            if state == "prompt":
                finish("FAIL: face lost")
        else:
            face = faces[0]
            live = spoof.score(frame, face.bbox)
            all_scores.append(live)
            if state == "prompt":
                round_scores.append(live)

            event = detector.update(face.landmark_2d_106, now)
            if event:
                last_blink = now

            if state == "wait":
                quiet = not detector.closed and now - last_blink >= QUIET
                if now >= prompt_due and quiet:
                    state = "prompt"
                    prompt_at = now
                    round_scores = [live]
                    print(f"round {round_no}: BLINK NOW")

            elif state == "prompt":
                if event and event.started >= prompt_at:
                    reaction = event.started - prompt_at
                    round_live = statistics.median(round_scores)
                    print(f"  reaction {reaction*1000:.0f} ms"
                          f"  blink {event.duration*1000:.0f} ms"
                          f"  scale {event.scale:.2f}  move {event.move:.2f}"
                          f"  live {round_live:.2f} (min {min(round_scores):.2f})"
                          f"  [{event.reason}]")
                    if not event.ok:
                        finish(f"FAIL: bad blink ({event.reason})")
                    elif reaction < MIN_REACTION:
                        finish("FAIL: too fast")
                    elif reaction > WINDOW:
                        finish("FAIL: too late")
                    elif round_live < LIVE_T:
                        finish(f"FAIL: spoof suspected (live {round_live:.2f})")
                    elif round_no == ROUNDS:
                        overall = statistics.median(all_scores)
                        if overall < LIVE_T:
                            finish(f"FAIL: spoof suspected (overall {overall:.2f})")
                        else:
                            finish(f"PASS: live (overall {overall:.2f},"
                                   f" min {min(all_scores):.2f})")
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

    view = cv2.flip(frame, 1)
    cv2.putText(view, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(view, f"relative {detector.rel:.3f}   live {live:.2f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    cv2.imshow("challenge", view)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# face-unlock

An experiment in Linux face unlock with real liveness detection on a **plain RGB webcam**, built to resist the photo and video spoofing that Howdy is vulnerable to.

> **Status: paused.** The Python liveness pipeline works and is documented below. The Rust PAM module (phase 4) was never started. The main finding is at the end: RGB-only liveness works in good light but can't be made reliable in the dark, which is likely why nobody ships serious face unlock without an IR camera. Contributions and forks are welcome; see [Picking this up](#picking-this-up).

## What works

`challenge.py` runs a randomized blink challenge. Every frame goes through three independent checks:

| Check | How | File |
|---|---|---|
| **Liveness (active)** | Random "BLINK NOW" prompts (3 rounds, random delay drawn with `secrets`). A blink must start 0.15–1.0 s after the prompt, last 0.10–0.8 s, and the face must stay still while the eyes are closed. A 1 s quiet period before each prompt stops a video of rapid blinking from getting lucky. | `liveness.py`, `challenge.py` |
| **Anti-spoof (passive)** | [facenox MiniFAS-V2-SE](https://github.com/facenox/face-antispoof-onnx) (Apache-2.0, quantized ONNX) scores whether the face is real skin or a screen or print. Each round's median score must be ≥ 0.5. | `antispoof.py` |
| **Identity** | InsightFace MobileFaceNet (`w600k_mbf`, `buffalo_sc` pack) against an enrolled template. Median similarity must be ≥ 0.5 each round. Any checked frame below 0.25 fails immediately. Identity is checked every 5th frame, plus on every blink frame, at each prompt, and whenever the face jumps (to catch a swap). | `identity.py` |

Everything fails closed: a missing model, a hash mismatch, a lost face, a timeout, or any failed check means deny.

### Integrity checks
- `antispoof.py` pins the model's SHA-256 and refuses to start if the file differs. The file was verified against the publisher's git blob hash (`a729f76b…`) via the GitHub API.
- The template (`enroll/naved.npz`) records the SHA-256 of the recognition model that created it. `identity.py` refuses a template made with a different model, and loads it with `allow_pickle=False`.

### Measured results (one laptop webcam, one user)

| Test | Result |
|---|---|
| Real face, normal indoor light | **PASS**, anti-spoof median 0.87–0.95, identity 0.82–0.89, ~9.6 fps |
| Real face, backlit | PASS, anti-spoof ≥ 0.93 |
| Phone photo / phone video / phone held close | anti-spoof **0.00–0.02** on every frame |
| Tilted photo (motion-blur fake blinks) | rejected by the minimum blink length + stability check |
| Looping blink video | never prompted (quiet period), times out |
| **Real face, low light** | **rejected**: anti-spoof 0.03–0.07 (fails closed) |

For comparison: a phone photo of the user matched plain face recognition at 0.86 against 0.93 live, which is why recognition alone (as in Howdy) isn't enough.

## Known limits (the honest part)

1. **Low light breaks the anti-spoof model.** In a dim room the webcam raises its gain, and the noisy image scores like a screen. It fails closed (your password still works), but face unlock is useless at night.
   - Tried: **screen as fill light** (fullscreen white window at maximum brightness). It didn't help (0.04). `FILL_LIGHT` in `challenge.py`, off by default.
   - Tried: **detecting "too dark" from face brightness.** It doesn't work, because auto-exposure normalizes the face brightness (dark runs measured ~115, brighter than normal light at ~99).
2. **Not covered:** 3D masks, and real-time deepfake "puppets" shown on a good screen. These aren't solvable with a single RGB camera. It's why Windows Hello requires an IR camera.
3. **Tested on one person, one webcam, one room.** The thresholds (`LIVE_T`, `MATCH_T`, blink timing) come from that data only.
4. **Performance.** On the test laptop, a webcam in dim light drops to ~5 fps (auto-exposure), which is too slow to catch blinks reliably. Face recognition with `buffalo_l`'s ResNet-50 took ~400 ms per frame on CPU, which is why MobileFaceNet is used instead (~20 ms).

**Conclusion:** without IR, face unlock can be made convincing in good light but not trustworthy in all conditions. On IR hardware, Howdy (plus `linux-enable-ir-emitter`) is the better starting point.

## Files

| File | Purpose |
|---|---|
| `enroll.py` | Capture 5 samples (press `c`), save `enroll/naved.npz` |
| `challenge.py` | The full liveness + identity challenge |
| `liveness.py` | Blink detector (106-point landmarks, hysteresis, stability check) |
| `antispoof.py` | Passive anti-spoof model wrapper with a pinned hash |
| `identity.py` | MobileFaceNet loader, template save/load, model-hash binding |
| `spoof_probe.py` | Side-by-side anti-spoof model comparison; `python spoof_probe.py <label>` prints min/p5/mean/max |
| `blink.py`, `eyes.py`, `detect.py`, `verify.py` | Earlier diagnostic steps (`verify.py` expects the old `.npy` template and is retired) |

`models/` and `enroll/` are git-ignored: the models are large, the `buffalo_*` weights are licensed for non-commercial use only, and enrolled faces must never be committed.

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install opencv-python insightface onnxruntime numpy
mkdir -p models
curl -fL -o models/facenox_best.onnx \
  https://raw.githubusercontent.com/facenox/face-antispoof-onnx/main/models/best_model_quantized.onnx
sha256sum models/facenox_best.onnx   # must match MODEL_SHA256 in antispoof.py
python enroll.py
python challenge.py
```

The InsightFace `buffalo_l` and `buffalo_sc` packs download automatically on first run.

## Roadmap

1. [x] Kali VM (VMware) with webcam passthrough + clean snapshot
2. [x] Standalone Python face match (enroll / verify)
3. [x] Liveness: randomized blink challenge + passive anti-spoof + per-round identity
4. [ ] Rust PAM module wired to the Python service (not started)

## Picking this up

Worthwhile next steps, roughly in order:

- **Fix low light.** This is the blocker. Ideas: read the camera's gain and exposure (`v4l2-ctl`) to detect darkness honestly and say "too dark, use password"; try other anti-spoof models or fine-tune one on noisy webcam images; lock the camera's frame rate (`exposure_dynamic_framerate`) and measure what that does to the scores.
- **Make it a service.** Wrap `challenge.py` as a daemon with no UI. Run it as an unprivileged user, listening on a Unix socket, returning allow/deny, with the template and models readable only by that user.
- **Phase 4: PAM module.** A small, fail-closed Rust module (`pam-bindings`) that asks the service and treats any error or timeout as deny, so PAM falls back to the password. Test only inside a VM, and snapshot before touching `/etc/pam.d/`.
- **More data.** Different faces, webcams, glasses and skin tones before trusting any threshold.

## License

MIT (this code). Models belong to their authors: facenox MiniFAS is Apache-2.0; InsightFace `buffalo_*` weights are for non-commercial research only.

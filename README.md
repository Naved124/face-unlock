# face-unlock

Linux face-unlock authentication with real liveness detection — built to resist the photo/video spoofing that Howdy is vulnerable to.

## Architecture

- **Rust PAM module**: tiny, fail-closed; asks the service over a Unix socket and returns allow/deny.
- **Python service**: runs as an unprivileged user; face matching (InsightFace/ArcFace) plus liveness (random challenge + passive anti-spoof).

## Roadmap

1. [x] Kali VM (VMware) with webcam passthrough + clean snapshot
2. [ ] Standalone Python face-match loop (enroll / verify / print distance)
3. [ ] Liveness detection (randomized blink/head-turn challenge, passive anti-spoof model)
4. [ ] Rust PAM module wired to the Python service

> PAM testing happens only inside the VM. Snapshot before touching `/etc/pam.d/`.

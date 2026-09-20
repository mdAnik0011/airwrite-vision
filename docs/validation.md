# Local validation — 2026-09-19

The implementation was checked on Windows with Python 3.11.9. These results
describe this run, not a guarantee for other machines or air-writing styles.

## Automated checks

- 26 camera-free tests passed, including a synthetic training/checkpoint/evaluation
  round trip and runtime inference.
- Ruff lint and formatting passed; `pip check` found no broken requirements.
- The official MediaPipe Hand Landmarker loaded, processed a blank frame and closed.
- Camera capture delivered a 640×480 BGR frame when run outside the sandbox.
  The restricted execution environment could not access the camera.
- The full application opened its window, processed 30 live camera frames and
  exited automatically with clean resource shutdown. All 30 frames had no detected
  hand, so this verifies startup/capture/rendering/shutdown, not live gesture quality.
- The upright EMNIST sample grid was visually inspected. Mixed upper/lowercase
  handwriting is expected in the Letters split.

## Measured baseline

Command: `python -m airwrite.ml.train --epochs 3 --batch-size 128`

| Setting / result | Value |
| --- | --- |
| Seed | 42 |
| Optimizer | AdamW, learning rate 0.001 |
| Device | CPU |
| Training samples | 112,320 |
| Validation samples | 12,480 |
| Best epoch | 3 |
| Best validation accuracy | 93.742% |
| Official test samples | 20,800 |
| Official test accuracy | 93.361% |
| Custom air-writing evaluation | Not performed |

Evaluation command: `python -m airwrite.ml.evaluate`. Per-class accuracy and the
complete confusion matrix are in `outputs/predictions/evaluation.json` locally.
The CNN checkpoint and training history are in `models/`. Generated assets are
Git-ignored and must be regenerated when cloning the source repository.

Checkpoint SHA-256:
`02d50070720d58cfbeaa55337bc315ce89a5d1bebf9a05998f43e4b3d6e678ac`

Runtime inference also loaded this checkpoint successfully: the first official
test sample (A) produced A as the top result with a 96.89% softmax score. This is
a single integration check, not an additional accuracy estimate.

## Remaining hands-on acceptance

A person must still check fingertip alignment, comfortable pinch thresholds,
toolbar hit targets, writing/erasing in motion, and recognition of their actual
air-written characters. No air-writing accuracy or sustained live FPS claim is
made. Use the custom collector and a separate writer/session holdout to measure
those outcomes.

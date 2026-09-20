# Model assets

Run `python scripts/download_models.py` from the repository root to install
`hand_landmarker.task`. This is the official version 1 float16 Hand Landmarker
asset from Google; `--force` replaces it. The downloader writes a temporary file
before replacing the target.

Train `airwrite_cnn.pt` with `python -m airwrite.ml.train --epochs 10` after
installing `.[ml]`. Checkpoints contain tensor weights, the A–Z class mapping,
preprocessing version, seed and best validation metrics. Only load checkpoints
you trust. Runtime uses PyTorch's tensor-only loader.

Drawing works without the CNN. The hand model is required for live tracking.
Downloaded assets, datasets and generated weights are ignored by Git; the
repository's MIT license covers its code, not third-party data or model assets.

References: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker),
[EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset).

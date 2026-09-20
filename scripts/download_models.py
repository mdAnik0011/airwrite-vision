"""Download the official MediaPipe Hand Landmarker task atomically."""

import argparse
from pathlib import Path
from urllib.request import urlopen

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def main():
    """Keep existing assets unless --force is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("models/hand_landmarker.task"))
    args = parser.parse_args()
    if args.output.is_file() and not args.force:
        print(f"Already installed: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".download")
    try:
        with urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded model is unexpectedly small; retry the download.")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Installed: {args.output}")


if __name__ == "__main__":
    main()

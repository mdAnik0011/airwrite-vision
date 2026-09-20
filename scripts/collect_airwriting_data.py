"""Collect labeled samples through the normal AirWrite UI."""

import argparse
from pathlib import Path
from string import ascii_uppercase

from airwrite.app import AirWriteApp
from airwrite.collection import SampleCollector
from airwrite.config import AppConfig


def main():
    """SAVE writes a PNG/JSON pair and clears for the next sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=list(ascii_uppercase))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/custom"))
    args = parser.parse_args()
    config = AppConfig()
    config.camera.camera_index = args.camera
    try:
        AirWriteApp(config, SampleCollector(args.label, args.output)).run()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Collector: {error}\n")


if __name__ == "__main__":
    main()

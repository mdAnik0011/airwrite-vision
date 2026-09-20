"""Check mirrored camera capture without requiring either model."""

import argparse

import cv2

from airwrite.camera import Camera
from airwrite.config import CameraConfig


def main():
    """Display frames until Q or window close."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    try:
        with Camera(CameraConfig(camera_index=args.camera)) as camera:
            while True:
                cv2.imshow("Camera test - Q to exit", camera.read())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if cv2.getWindowProperty("Camera test - Q to exit", cv2.WND_PROP_VISIBLE) < 1:
                    break
    except RuntimeError as error:
        parser.exit(1, f"Camera: {error}\n")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

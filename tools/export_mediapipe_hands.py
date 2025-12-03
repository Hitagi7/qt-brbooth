"""Export MediaPipe's hand landmark TFLite model to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapipe as mp
from tflite2onnx import convert


PACKAGE_DIR = Path(mp.__file__).resolve().parent
MODEL_DIR = PACKAGE_DIR / "modules" / "hand_landmark"

MODEL_FILES = {
    "lite": "hand_landmark_lite.tflite",
    "full": "hand_landmark_full.tflite",
}


def export_hand_landmark(output_path: Path, model_type: str) -> None:
    try:
        model_name = MODEL_FILES[model_type]
    except KeyError as exc:
        raise ValueError("model_type must be one of: {}".format(", ".join(MODEL_FILES.keys()))) from exc

    tflite_path = MODEL_DIR / model_name
    if not tflite_path.exists():
        raise FileNotFoundError(f"Could not locate {tflite_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    convert(str(tflite_path), str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_FILES.keys(), default="full")
    args = parser.parse_args()

    export_hand_landmark(args.output, args.model)


if __name__ == "__main__":
    main()


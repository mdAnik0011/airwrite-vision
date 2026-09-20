"""Optional runtime inference; missing ML dependencies never prevent drawing."""

from dataclasses import dataclass
from pathlib import Path
from pickle import UnpicklingError

from airwrite.preprocessing import preprocess_canvas


@dataclass(frozen=True)
class Prediction:
    """Letter and softmax score (not a calibrated probability)."""

    character: str
    confidence: float


class CharacterRecognizer:
    """Load once on CPU and perform top-three inference only on request."""

    def __init__(self, model_path: Path):
        self.model = None
        self.status = "Recognition model not installed/trained."
        self.unavailable_reason = self.status
        if not model_path.is_file():
            return
        try:
            import torch

            from airwrite.ml.model import CLASSES, load_checkpoint

            self.torch = torch
            self.classes = CLASSES
            self.model = load_checkpoint(model_path).eval()
            self.status = "Recognition ready."
        except (
            ImportError,
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            EOFError,
            UnpicklingError,
            IndexError,
            TypeError,
            AttributeError,
        ) as error:
            self.status = f"Recognition unavailable: {error}"
            self.unavailable_reason = self.status

    def predict(self, canvas, top_k: int = 3) -> list[Prediction]:
        """Return predictions or an actionable status for missing model/empty ink."""
        image = preprocess_canvas(canvas)
        if image is None:
            self.status = "Nothing to recognize."
            return []
        if self.model is None:
            self.status = self.unavailable_reason
            return []
        with self.torch.inference_mode():
            logits = self.model(self.torch.from_numpy(image).unsqueeze(0))
            values, indices = logits.softmax(1).topk(max(1, min(top_k, 26)))
        self.status = "Recognition complete."
        return [
            Prediction(self.classes[i], float(value))
            for i, value in zip(indices[0].tolist(), values[0].tolist(), strict=True)
        ]

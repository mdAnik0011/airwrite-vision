"""Small synthetic checks verify ML plumbing, not handwriting accuracy."""

import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from airwrite.ml import evaluate, train  # noqa: E402
from airwrite.ml.dataset import LetterTransform, zero_based_label  # noqa: E402
from airwrite.preprocessing import preprocess_canvas  # noqa: E402
from airwrite.recognizer import CharacterRecognizer  # noqa: E402


def test_emnist_orientation_and_labels():
    upright = np.zeros((28, 28), dtype=np.uint8)
    upright[3:24, 6:10] = 255
    upright[20:24, 6:23] = 255
    transformed = LetterTransform()(Image.fromarray(upright.T))
    assert np.array_equal(transformed.numpy(), preprocess_canvas(upright))
    assert zero_based_label(1) == 0 and zero_based_label(26) == 25


def test_train_checkpoint_evaluate_and_runtime(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    dataset = torch.utils.data.TensorDataset(torch.rand(20, 1, 28, 28), torch.arange(20) % 26)
    monkeypatch.setattr(train, "emnist", lambda _: dataset)
    monkeypatch.setattr(train, "save_orientation_preview", lambda *_: None)
    monkeypatch.setattr(evaluate, "emnist", lambda *_, **__: dataset)
    checkpoint = tmp_path / "model.pt"
    train.main(["--epochs", "1", "--batch-size", "8", "--output", str(checkpoint)])
    assert checkpoint.is_file()
    history = json.loads(checkpoint.with_suffix(".history.json").read_text())
    assert len(history) == 1 and history[0]["train_loss"] > 0
    results = tmp_path / "evaluation.json"
    evaluate.main(["--model", str(checkpoint), "--output", str(results)])
    report = json.loads(results.read_text())
    assert report["samples"] == 20
    assert sum(map(sum, report["confusion_matrix"])) == 20
    recognizer = CharacterRecognizer(checkpoint)
    image = np.zeros((50, 50), dtype=np.uint8)
    image[10:40, 20:25] = 255
    predictions = recognizer.predict(image)
    assert len(predictions) == 3
    assert all(0 <= p.confidence <= 1 for p in predictions)
    assert predictions[0].confidence >= predictions[-1].confidence


def test_corrupt_model_keeps_drawing_available(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"broken checkpoint")
    recognizer = CharacterRecognizer(path)
    assert recognizer.model is None
    assert "unavailable" in recognizer.status

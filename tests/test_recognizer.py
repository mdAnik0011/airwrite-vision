import numpy as np

from airwrite.recognizer import CharacterRecognizer


def test_missing_model_and_empty_canvas(tmp_path):
    recognizer = CharacterRecognizer(tmp_path / "missing.pt")
    blank = np.zeros((50, 50), dtype=np.uint8)
    assert recognizer.predict(blank) == []
    assert recognizer.status == "Nothing to recognize."
    blank[10:20, 10:20] = 255
    assert recognizer.predict(blank) == []
    assert recognizer.status == "Recognition model not installed/trained."

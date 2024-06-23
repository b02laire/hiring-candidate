from backend_exercise import DocTableDetector
import pytest
from PIL import UnidentifiedImageError


def test_invalid_format():
    tab_detector = DocTableDetector()
    with pytest.raises(UnidentifiedImageError):
        pred = tab_detector.predict("test_samples/expense_report.pdf")


def test_success():
    """Verify the output to a known image returns results"""
    tab_detector = DocTableDetector()
    y = tab_detector.predict("test_samples/expense_report.png")
    y = tab_detector.format_predictions(y)
    assert y


def test_failure(monkeypatch):
    """Simulates an error during image-preprocessing"""
    tab_detector = DocTableDetector()

    def mock_preprocess_image(*args, **kwargs):
        raise ValueError("Mocked error in preprocess_image")

    monkeypatch.setattr(tab_detector, "preprocess_image",
                        mock_preprocess_image)

    with pytest.raises(ValueError):
        tab_detector.predict("test_samples/expense_report.png")


def test_blank_image():
    """Verify the output of a white image is empty"""
    tab_detector = DocTableDetector()
    y = tab_detector.predict("test_samples/white.png")
    y = tab_detector.format_predictions(y)
    assert not y

def test_empty_file():
    """Verify the Detector's reaction to an empty file"""
    tab_detector = DocTableDetector()
    with pytest.raises(UnidentifiedImageError):
        tab_detector.predict("test_samples/empty.png")

@pytest.mark.parametrize("image_path", [
    "test_samples/expense_report.png",
    "test_samples/expense_report.jpg",
    "test_samples/expense_report.bmp"])
def test_different_image_formats(image_path):
    """Verify the predictions to different image formats match each other"""
    tab_detector = DocTableDetector()
    y = tab_detector.predict(image_path)
    y = tab_detector.format_predictions(y)
    print(y)
    assert 0.994 <= y[0].get("score") <= 0.995


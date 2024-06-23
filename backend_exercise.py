from typing import List, Dict

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image


class DocTableDetector:
    """Handles table detection on invoice and bank document image"""

    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained(
            "TahaDouaji/detr-doc-table-detection")
        self.model = DetrForObjectDetection.from_pretrained(
            "TahaDouaji/detr-doc-table-detection")

    @staticmethod
    def preprocess_image(image_path: str) -> Image:
        """Opens an image and converts it for use by the model

        Args:
            image_path: path of the image to preprocess
        """
        return Image.open(image_path).convert("RGB")

    def predict(self, image_path: str, pred_threshold: float = 0.9) -> List[
        Dict]:
        """Detects tables within an image and returns a dict for each detected
        table containing the detection confidence and the box coordinates.

        Args:
            image_path: path of the image to predict on
            pred_threshold: min confidence level of the prediction to return
        """
        image = self.preprocess_image(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image.size[::-1]]),
            threshold=pred_threshold
        )

        return results

    @staticmethod
    def format_predictions(predictions: List[Dict]) -> List[Dict]:
        """Formats the predictions in a human-readable way"""
        formated_preds = []
        for pred in predictions:
            for score, label, box in zip(pred["scores"], pred["labels"],
                                         pred["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                formated_preds.append(dict(score=round(score.item(), 3),
                                           box=box))
        return formated_preds

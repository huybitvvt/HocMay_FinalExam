from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from object_detection import draw_detections, parse_yolo_result


class FakeBoxes:
    def __init__(self):
        self.xyxy = np.array([[10, 20, 100, 120], [30, 40, 80, 90]], dtype=float)
        self.conf = np.array([0.91, 0.72], dtype=float)
        self.cls = np.array([8, 6], dtype=float)

    def __len__(self):
        return len(self.conf)


class FakeResult:
    boxes = FakeBoxes()
    names = {8: "plastic", 6: "metal"}


class ObjectDetectionTests(unittest.TestCase):
    def test_parse_yolo_result(self):
        detections = parse_yolo_result(FakeResult())

        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["class_name"], "plastic")
        self.assertEqual(detections[0]["label"], "Nhựa")
        self.assertAlmostEqual(detections[0]["confidence"], 0.91)

    def test_draw_detections_preserves_image_size(self):
        image = Image.new("RGB", (160, 160), "white")
        detections = parse_yolo_result(FakeResult())
        annotated = draw_detections(image, detections)

        self.assertEqual(annotated.size, image.size)
        self.assertNotEqual(annotated.getpixel((10, 20)), image.getpixel((10, 20)))


if __name__ == "__main__":
    unittest.main()

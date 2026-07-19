from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from cleanliness_model import cleanliness_advice, predict_cleanliness


class FakeModel:
    input_shape = (None, 224, 224, 3)

    def predict(self, batch, verbose=0):
        return np.array([[0.10, 0.82, 0.08]], dtype=float)


class CleanlinessModelTests(unittest.TestCase):
    def test_predict_cleanliness(self):
        image = Image.new("RGB", (320, 240), "white")
        result = predict_cleanliness(FakeModel(), image)

        self.assertEqual(result["class_name"], "dirty")
        self.assertEqual(result["label"], "Có thể bị bẩn")
        self.assertAlmostEqual(result["confidence"], 0.82)

    def test_advice_only_applies_to_recyclable_classes(self):
        result = {"class_name": "dirty", "confidence": 0.82}
        recyclable_advice = cleanliness_advice("plastic", result)
        other_advice = cleanliness_advice("biological", result)

        self.assertIn("rửa sơ", recyclable_advice)
        self.assertIn("Không áp dụng", other_advice)


if __name__ == "__main__":
    unittest.main()

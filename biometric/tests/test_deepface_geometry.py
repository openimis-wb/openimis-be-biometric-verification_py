"""
DeepFace face geometry for the quality gate (docs/wb-biometric-dedup-seam.md §6.7).

The dicts below follow the shape DeepFace.represent() returns: one dict per
face with "embedding", "facial_area" and "face_confidence". facial_area holds
x, y, w, h and the eye points on every supported version; nose, mouth_left
and mouth_right only on versions that pass them through, and only for
detectors that report them (retinaface). A landmark the detector did not find
is None. DeepFace is not imported: extract() is exercised with it patched.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import SimpleTestCase

from biometric.providers.base import FaceGeometry
from biometric.providers.deepface_provider import DeepFaceProvider, face_geometry_from_deepface
from biometric.quality import assess, pillow_available

RETINAFACE_RESULT = {
    "embedding": [0.1, 0.2, 0.3, 0.4],
    "facial_area": {
        "x": 30, "y": 30, "w": 60, "h": 60,
        "left_eye": (75, 50), "right_eye": (45, 50),
        "nose": (60, 65), "mouth_left": (72, 80), "mouth_right": (48, 80),
    },
    "face_confidence": 0.99,
}

OPENCV_ONE_EYE_RESULT = {
    "embedding": [0.1, 0.2, 0.3, 0.4],
    "facial_area": {"x": 12, "y": 8, "w": 40, "h": 44, "left_eye": None, "right_eye": (20, 22)},
    "face_confidence": 0.87,
}

# enforce_detection=False and no face found: the whole frame (clamped to
# width - 1 / height - 1), confidence 0, no eye.
NO_DETECTION_RESULT = {
    "embedding": [0.1, 0.2, 0.3, 0.4],
    "facial_area": {"x": 0, "y": 0, "w": 119, "h": 119, "left_eye": None, "right_eye": None},
    "face_confidence": 0,
}

# detector_backend="skip": a dummy region over the whole frame.
SKIP_RESULT = {
    "embedding": [0.1, 0.2, 0.3, 0.4],
    "facial_area": {"x": 0, "y": 0, "w": 120, "h": 120},
    "face_confidence": 0,
}


class TestFaceGeometryFromDeepFace(SimpleTestCase):

    def test_retinaface_result_gives_box_and_five_landmarks(self):
        face = face_geometry_from_deepface(RETINAFACE_RESULT, detector_backend="retinaface")

        self.assertEqual(face.box, (30.0, 30.0, 60.0, 60.0))
        self.assertEqual(face.landmarks, {
            "left_eye": (75.0, 50.0), "right_eye": (45.0, 50.0), "nose": (60.0, 65.0),
            "mouth_left": (72.0, 80.0), "mouth_right": (48.0, 80.0),
        })
        self.assertIsNone(face.pose)

    def test_missing_landmark_is_skipped(self):
        face = face_geometry_from_deepface(OPENCV_ONE_EYE_RESULT, detector_backend="opencv")

        self.assertEqual(face.box, (12.0, 8.0, 40.0, 44.0))
        self.assertEqual(face.landmarks, {"right_eye": (20.0, 22.0)})

    def test_list_points_and_numpy_scalars_are_read(self):
        result = {
            "facial_area": {
                "x": np.int64(1), "y": np.int64(2), "w": np.int64(3), "h": np.int64(4),
                "left_eye": [np.int32(5), np.int32(6)], "right_eye": [7, 8],
            },
            "face_confidence": 0.5,
        }
        face = face_geometry_from_deepface(result, detector_backend="retinaface")

        self.assertEqual(face.box, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(face.landmarks, {"left_eye": (5.0, 6.0), "right_eye": (7.0, 8.0)})
        self.assertIsInstance(face.box[0], float)

    def test_skip_detector_gives_no_geometry(self):
        self.assertIsNone(face_geometry_from_deepface(SKIP_RESULT, detector_backend="skip"))

    def test_no_detection_fallback_gives_no_geometry(self):
        self.assertIsNone(face_geometry_from_deepface(NO_DETECTION_RESULT, detector_backend="opencv"))

    def test_missing_or_malformed_facial_area_gives_no_geometry(self):
        for result in ({}, {"facial_area": None}, {"facial_area": "30,30,60,60"}, None):
            with self.subTest(result=result):
                self.assertIsNone(face_geometry_from_deepface(result, detector_backend="retinaface"))

    def test_invalid_box_is_dropped_and_landmarks_kept(self):
        for box in ({"x": 1, "y": 2, "w": 0, "h": 4}, {"x": "a", "y": 2, "w": 3, "h": 4},
                    {"x": 1, "y": 2, "w": float("nan"), "h": 4}, {"x": 1, "y": 2, "h": 4}):
            with self.subTest(box=box):
                result = {"facial_area": {**box, "left_eye": (5, 6), "right_eye": (9, 6)}, "face_confidence": 0.9}
                face = face_geometry_from_deepface(result, detector_backend="retinaface")
                self.assertIsNone(face.box)
                self.assertEqual(face.landmarks, {"left_eye": (5.0, 6.0), "right_eye": (9.0, 6.0)})

    def test_malformed_points_are_skipped(self):
        result = {
            "facial_area": {
                "x": 1, "y": 2, "w": 3, "h": 4,
                "left_eye": (1,), "right_eye": ("a", 2), "nose": (float("inf"), 1), "mouth_left": (3, 4),
                "chin": (1, 1),
            },
            "face_confidence": 0.9,
        }
        face = face_geometry_from_deepface(result, detector_backend="retinaface")

        self.assertEqual(face.landmarks, {"mouth_left": (3.0, 4.0)})


def _noise_png(size=120):
    from PIL import Image

    rng = np.random.default_rng(7)
    buffer = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestDeepFaceProviderExtract(SimpleTestCase):

    def _extract(self, result, detector_backend="retinaface"):
        fake = MagicMock()
        fake.represent.return_value = [result]
        with patch("biometric.providers.deepface_provider._DeepFace", fake), \
                patch("biometric.providers.deepface_provider._DEEPFACE_AVAILABLE", True):
            provider = DeepFaceProvider(detector_backend=detector_backend)
            return provider.extract(_noise_png()), fake

    def test_extract_returns_the_embedding_and_the_geometry_of_the_same_face(self):
        extracted, fake = self._extract(RETINAFACE_RESULT)

        self.assertEqual(extracted.vector, [0.1, 0.2, 0.3, 0.4])
        self.assertIsInstance(extracted.face, FaceGeometry)
        self.assertEqual(extracted.face.box, (30.0, 30.0, 60.0, 60.0))
        self.assertIsNone(extracted.quality)
        _, kwargs = fake.represent.call_args
        self.assertEqual(kwargs["detector_backend"], "retinaface")

    def test_extract_with_skip_detector_has_no_geometry(self):
        extracted, _ = self._extract(SKIP_RESULT, detector_backend="skip")

        self.assertEqual(extracted.vector, [0.1, 0.2, 0.3, 0.4])
        self.assertIsNone(extracted.face)

    def test_quality_gate_judges_roll_and_measures_the_lower_face(self):
        if not pillow_available():
            self.skipTest("Pillow is needed to decode the sample")
        extracted, _ = self._extract(RETINAFACE_RESULT)

        verdict = assess("face", _noise_png(), extracted, server_extracted=True, mode_value="advisory")
        measures = {m.name: m for m in verdict.measures}

        self.assertEqual(measures["roll"].source, "landmarks")
        self.assertEqual(measures["roll"].value, 0.0)
        self.assertTrue(measures["roll"].passed)
        self.assertNotEqual(measures["roll"].detail, "no_pose")
        self.assertIsNotNone(measures["lower_face_uniformity"].value)
        self.assertNotEqual(measures["lower_face_uniformity"].detail, "no_face_box")
        self.assertIsNotNone(measures["yaw_ratio"].value)
        self.assertEqual(measures["yaw"].detail, "no_pose")

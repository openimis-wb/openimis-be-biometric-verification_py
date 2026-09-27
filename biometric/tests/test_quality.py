"""
Unit tests for biometric/quality.py (§6.7): the pure measures, the verdict
status rules and config resolution. Images are built with NumPy and encoded
through Pillow; image tests are skipped when Pillow is not importable.
"""

import io
import json
import math
from unittest import skipUnless
from unittest.mock import patch

import numpy as np
from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase

from biometric import apps as biometric_apps
from biometric.apps import BiometricConfig
from biometric.providers.base import Extracted, FaceGeometry
from biometric.quality import (
    ACCEPTED,
    NOT_ASSESSED,
    REFUSED,
    QualityMeasure,
    QualityRefusedError,
    assess,
    decode_image,
    lower_face_uniformity,
    mode,
    pillow_available,
    pose_measures,
    sharpness,
    srgb_to_lab,
    thresholds_for,
)

FACE_DEFAULTS = {
    "min_sharpness": 100.0,
    "max_yaw": 20.0,
    "max_pitch": None,
    "max_roll": 20.0,
    "max_yaw_ratio": None,
    "min_lower_face_uniformity": None,
    "min_quality": None,
}


def png(array) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def sharp_png(size=64, seed=7) -> bytes:
    rng = np.random.default_rng(seed)
    return png(rng.integers(0, 256, size=(size, size), dtype=np.uint8))


def blurry_png(size=64) -> bytes:
    return png(np.full((size, size), 128, dtype=np.uint8))


def _measure(measures, name):
    return next(m for m in measures if m.name == name)


class _QualityConfigTestCase(SimpleTestCase):
    """Restores BiometricConfig.quality after each test."""

    def setUp(self):
        super().setUp()
        self._quality = BiometricConfig.quality

    def tearDown(self):
        BiometricConfig.quality = self._quality
        super().tearDown()


class TestSharpness(SimpleTestCase):

    @staticmethod
    def _reference(gray):
        """Per-pixel 5-point stencil with an explicit mirror-without-edge border."""
        h, w = len(gray), len(gray[0])

        def at(i, j):
            i = -i if i < 0 else (2 * (h - 1) - i if i >= h else i)
            j = -j if j < 0 else (2 * (w - 1) - j if j >= w else j)
            return float(gray[i][j])

        values = [
            at(i - 1, j) + at(i + 1, j) + at(i, j - 1) + at(i, j + 1) - 4.0 * at(i, j)
            for i in range(h) for j in range(w)
        ]
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def test_constant_image_is_zero(self):
        self.assertEqual(sharpness(np.full((10, 10), 200.0)), 0.0)

    def test_matches_per_pixel_reference_on_4x4(self):
        gray = np.array([
            [10, 20, 30, 40],
            [50, 0, 70, 80],
            [90, 100, 255, 120],
            [130, 140, 150, 5],
        ], dtype=np.float32)
        self.assertAlmostEqual(sharpness(gray), self._reference(gray.tolist()), places=3)

    def test_noise_is_sharp_and_flat_is_not(self):
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 256, size=(64, 64)).astype(np.float32)
        self.assertGreater(sharpness(noise), 100.0)
        self.assertEqual(sharpness(np.full((64, 64), 128.0)), 0.0)

    def test_smaller_than_3x3_is_zero(self):
        self.assertEqual(sharpness(np.array([[0.0, 255.0], [255.0, 0.0]])), 0.0)
        self.assertEqual(sharpness(np.zeros((2, 10))), 0.0)


@skipUnless(pillow_available(), "Pillow is not importable")
class TestDecodeImage(SimpleTestCase):

    def test_decodes_png(self):
        gray, rgb, detail = decode_image(sharp_png(16))
        self.assertEqual(gray.shape, (16, 16))
        self.assertEqual(gray.dtype, np.float32)
        self.assertEqual(rgb.shape, (16, 16, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(detail, "")

    def test_not_an_image_is_undecodable(self):
        self.assertEqual(decode_image(b"not-an-image"), (None, None, "undecodable"))

    def test_pillow_missing(self):
        with patch("biometric.quality._pil", return_value=None):
            self.assertEqual(decode_image(sharp_png(16)), (None, None, "pillow_unavailable"))

    def test_decompression_bomb_is_undecodable(self):
        from PIL import Image

        with patch.object(Image, "open", side_effect=Image.DecompressionBombError("bomb")):
            self.assertEqual(decode_image(sharp_png(16)), (None, None, "undecodable"))


class TestSrgbToLab(SimpleTestCase):

    def test_white(self):
        lab = srgb_to_lab(np.array([[[255, 255, 255]]], dtype=np.uint8))[0, 0]
        self.assertAlmostEqual(lab[0], 100.0, delta=0.5)
        self.assertAlmostEqual(lab[1], 0.0, delta=0.5)
        self.assertAlmostEqual(lab[2], 0.0, delta=0.5)

    def test_black(self):
        lab = srgb_to_lab(np.array([[[0, 0, 0]]], dtype=np.uint8))[0, 0]
        for component in lab:
            self.assertAlmostEqual(component, 0.0, delta=0.5)


class TestLowerFaceUniformity(SimpleTestCase):

    SKIN = (180, 120, 90)

    def _flat(self, size=100):
        return np.tile(np.array(self.SKIN, dtype=np.uint8), (size, size, 1))

    def test_flat_region_is_near_zero(self):
        value, detail = lower_face_uniformity(self._flat(), FaceGeometry(box=(0, 0, 100, 100)))
        self.assertAlmostEqual(value, 0.0, places=6)
        self.assertEqual(detail, "")

    def test_shaded_lower_face_measures_higher_than_flat(self):
        shaded = self._flat().astype(np.int32)
        ramp = np.linspace(-60, 60, 100).astype(np.int32)[:, None, None]
        shaded = np.clip(shaded + ramp, 0, 255).astype(np.uint8)
        face = FaceGeometry(box=(0, 0, 100, 100))
        flat_value, _ = lower_face_uniformity(self._flat(), face)
        shaded_value, _ = lower_face_uniformity(shaded, face)
        self.assertGreater(shaded_value, flat_value + 1.0)

    def test_mouth_band_is_cut_when_both_corners_given(self):
        # Region rows 30..50, columns 20..80 (1200 px); the mouth band covers
        # rows 34..46, columns 23..77 (648 px), filled with noise.
        box = (0, 0, 100, 50)
        corners = {"mouth_left": (25.0, 40.0), "mouth_right": (75.0, 40.0)}
        with_lips = self._flat()
        rng = np.random.default_rng(3)
        with_lips[34:46, 23:77] = rng.integers(0, 256, size=(12, 54, 3), dtype=np.uint8)

        bare_value, _ = lower_face_uniformity(self._flat(), FaceGeometry(box=box, landmarks=corners))
        cut_value, _ = lower_face_uniformity(with_lips, FaceGeometry(box=box, landmarks=corners))
        uncut_value, _ = lower_face_uniformity(with_lips, FaceGeometry(box=box))

        self.assertAlmostEqual(cut_value, bare_value, places=6)
        self.assertGreater(uncut_value, cut_value + 1.0)

    def test_nose_landmark_sets_the_region_top(self):
        image = self._flat()
        rng = np.random.default_rng(11)
        image[:60] = rng.integers(0, 256, size=(60, 100, 3), dtype=np.uint8)
        # The default top (row 60) sees only skin; a nose at row 10 pulls the noisy rows in.
        default_value, _ = lower_face_uniformity(image, FaceGeometry(box=(0, 0, 100, 100)))
        nose_value, _ = lower_face_uniformity(
            image, FaceGeometry(box=(0, 0, 100, 100), landmarks={"nose": (50.0, 10.0)}),
        )
        self.assertAlmostEqual(default_value, 0.0, places=6)
        self.assertGreater(nose_value, 0.0)

    def test_tiny_box_is_too_small(self):
        self.assertEqual(
            lower_face_uniformity(self._flat(), FaceGeometry(box=(10, 10, 5, 5))), (None, "region_too_small"),
        )

    def test_box_partly_outside_is_clipped(self):
        value, detail = lower_face_uniformity(self._flat(), FaceGeometry(box=(50, 30, 100, 100)))
        self.assertIsNotNone(value)
        self.assertEqual(detail, "")

    def test_no_box(self):
        self.assertEqual(lower_face_uniformity(self._flat(), FaceGeometry()), (None, "no_face_box"))
        self.assertEqual(lower_face_uniformity(self._flat(), None), (None, "no_face_box"))


class TestPoseMeasures(SimpleTestCase):

    def _eyes(self, a, b, **more):
        return FaceGeometry(landmarks={"left_eye": a, "right_eye": b, **more})

    def test_level_eyes_roll_zero_passes(self):
        roll = _measure(pose_measures(self._eyes((40, 50), (60, 50)), FACE_DEFAULTS), "roll")
        self.assertEqual(roll.value, 0.0)
        self.assertTrue(roll.passed)
        self.assertEqual(roll.source, "landmarks")

    def test_tilted_eyes_fail_default_roll(self):
        roll = _measure(pose_measures(self._eyes((40, 50), (60, 60)), FACE_DEFAULTS), "roll")
        self.assertAlmostEqual(roll.value, math.degrees(math.atan2(10, 20)), places=6)
        self.assertAlmostEqual(roll.value, 26.57, places=2)
        self.assertFalse(roll.passed)

    def test_roll_at_the_bound_is_admitted(self):
        roll = _measure(pose_measures(FaceGeometry(pose={"roll": 20.0}), FACE_DEFAULTS), "roll")
        self.assertTrue(roll.passed)
        negative = _measure(pose_measures(FaceGeometry(pose={"roll": -20.0}), FACE_DEFAULTS), "roll")
        self.assertTrue(negative.passed)

    def test_swapped_eye_order_gives_same_roll(self):
        a = _measure(pose_measures(self._eyes((40, 50), (60, 60)), FACE_DEFAULTS), "roll").value
        b = _measure(pose_measures(self._eyes((60, 60), (40, 50)), FACE_DEFAULTS), "roll").value
        self.assertAlmostEqual(a, b, places=9)

    def test_provider_yaw_beyond_bound_fails(self):
        yaw = _measure(pose_measures(FaceGeometry(pose={"yaw": 25.0}), FACE_DEFAULTS), "yaw")
        self.assertFalse(yaw.passed)
        self.assertEqual(yaw.source, "provider_pose")

    def test_pitch_is_recorded_not_judged(self):
        pitch = _measure(pose_measures(FaceGeometry(pose={"pitch": -30.0}), FACE_DEFAULTS), "pitch")
        self.assertEqual(pitch.value, -30.0)
        self.assertIsNone(pitch.passed)

    def test_provider_pose_wins_over_landmark_roll(self):
        face = FaceGeometry(landmarks={"left_eye": (40, 50), "right_eye": (60, 60)}, pose={"roll": 3.0})
        roll = _measure(pose_measures(face, FACE_DEFAULTS), "roll")
        self.assertEqual(roll.value, 3.0)
        self.assertEqual(roll.source, "provider_pose")

    def test_missing_angles_are_recorded_as_no_pose(self):
        measures = pose_measures(None, FACE_DEFAULTS)
        for axis in ("yaw", "pitch", "roll"):
            m = _measure(measures, axis)
            self.assertIsNone(m.value)
            self.assertEqual(m.detail, "no_pose")
            self.assertIsNone(m.passed)

    def test_yaw_ratio_from_nose(self):
        centred = _measure(
            pose_measures(self._eyes((40, 50), (60, 50), nose=(50, 60)), FACE_DEFAULTS), "yaw_ratio",
        )
        self.assertAlmostEqual(centred.value, 0.0, places=9)
        self.assertIsNone(centred.passed)

        under_eye = _measure(
            pose_measures(self._eyes((40, 50), (60, 50), nose=(60, 70)), FACE_DEFAULTS), "yaw_ratio",
        )
        self.assertAlmostEqual(abs(under_eye.value), 1.0, places=9)
        self.assertIsNone(under_eye.passed)

        judged = _measure(
            pose_measures(self._eyes((40, 50), (60, 50), nose=(60, 70)), {**FACE_DEFAULTS, "max_yaw_ratio": 0.5}),
            "yaw_ratio",
        )
        self.assertFalse(judged.passed)

    def test_yaw_ratio_follows_the_eye_axis(self):
        # Eyes tilted 45 degrees; the nose sits on the perpendicular through their midpoint.
        face = self._eyes((40, 40), (60, 60), nose=(40, 60))
        ratio = _measure(pose_measures(face, FACE_DEFAULTS), "yaw_ratio")
        self.assertAlmostEqual(ratio.value, 0.0, places=9)

    def test_invalid_geometry_never_raises(self):
        for face in (
            self._eyes((50, 50), (50, 50), nose=(50, 60)),
            self._eyes((float("nan"), 50), (60, 50), nose=(50, 60)),
            self._eyes(("x",), (60, 50)),
        ):
            measures = pose_measures(face, FACE_DEFAULTS)
            roll = _measure(measures, "roll")
            self.assertIsNone(roll.value)
            self.assertEqual(roll.detail, "geometry_invalid")
            self.assertIsNone(roll.passed)

    def test_non_finite_provider_angle_is_not_judged(self):
        yaw = _measure(pose_measures(FaceGeometry(pose={"yaw": float("inf")}), FACE_DEFAULTS), "yaw")
        self.assertIsNone(yaw.value)
        self.assertEqual(yaw.detail, "geometry_invalid")


class TestQualityMeasure(SimpleTestCase):

    def test_passed_is_none_without_value_or_limit(self):
        self.assertIsNone(QualityMeasure("x", None, 1.0, "min", "image").passed)
        self.assertIsNone(QualityMeasure("x", 1.0, None, "min", "image").passed)

    def test_bounds_are_admitted(self):
        self.assertTrue(QualityMeasure("x", 100.0, 100.0, "min", "image").passed)
        self.assertFalse(QualityMeasure("x", 99.9, 100.0, "min", "image").passed)
        self.assertTrue(QualityMeasure("x", -20.0, 20.0, "max", "image").passed)
        self.assertFalse(QualityMeasure("x", -20.1, 20.0, "max", "image").passed)


@skipUnless(pillow_available(), "Pillow is not importable")
class TestAssessStatusRules(_QualityConfigTestCase):

    def test_sharp_face_without_geometry_is_accepted(self):
        verdict = assess("face", sharp_png(), Extracted(vector=[1.0]), mode_value="advisory")
        self.assertEqual(verdict.status, ACCEPTED)
        self.assertEqual(verdict.reasons, [])
        self.assertTrue(_measure(verdict.measures, "sharpness").passed)
        for axis in ("yaw", "pitch", "roll"):
            self.assertIsNone(_measure(verdict.measures, axis).value)

    def test_blurry_face_is_refused(self):
        verdict = assess("face", blurry_png(), Extracted(vector=[1.0]), mode_value="advisory")
        self.assertEqual(verdict.status, REFUSED)
        self.assertEqual(verdict.reasons, ["sharpness_below_min"])

    def test_undecodable_server_extracted_sample_is_refused(self):
        verdict = assess("face", b"face-a", Extracted(vector=[1.0]), server_extracted=True, mode_value="advisory")
        self.assertEqual(verdict.status, REFUSED)
        self.assertEqual(verdict.reasons, ["sample_undecodable"])

    def test_undecodable_device_sample_is_not_assessed(self):
        verdict = assess("face", b"face-a", Extracted(vector=[1.0]), server_extracted=False, mode_value="advisory")
        self.assertEqual(verdict.status, NOT_ASSESSED)
        self.assertEqual(_measure(verdict.measures, "sharpness").detail, "sample_not_image")
        self.assertEqual(_measure(verdict.measures, "lower_face_uniformity").detail, "sample_not_image")

    def test_without_pillow_face_is_not_assessed(self):
        with patch("biometric.quality._pil", return_value=None):
            verdict = assess("face", blurry_png(), Extracted(vector=[1.0]), mode_value="enforce")
        self.assertEqual(verdict.status, NOT_ASSESSED)
        self.assertEqual(_measure(verdict.measures, "sharpness").detail, "pillow_unavailable")

    def test_without_pillow_undecodable_server_sample_is_not_refused(self):
        with patch("biometric.quality._pil", return_value=None):
            verdict = assess("face", b"face-a", Extracted(vector=[1.0]), mode_value="enforce")
        self.assertEqual(verdict.status, NOT_ASSESSED)

    def test_reported_quality_without_limit_is_recorded(self):
        verdict = assess("fingerprint", b"tmpl", Extracted(template=b"tmpl", quality=30.0), mode_value="advisory")
        self.assertEqual(verdict.status, NOT_ASSESSED)
        self.assertEqual([m.name for m in verdict.measures], ["provider_quality"])
        self.assertEqual(verdict.measures[0].value, 30.0)

    def test_reported_quality_below_configured_min_is_refused(self):
        BiometricConfig.quality = {"mode": "advisory", "modalities": {"fingerprint": {"min_quality": 40}}}
        verdict = assess("fingerprint", b"tmpl", Extracted(template=b"tmpl", quality=30.0))
        self.assertEqual(verdict.status, REFUSED)
        self.assertEqual(verdict.reasons, ["provider_quality_below_min"])

    def test_unreported_quality_with_min_is_not_assessed(self):
        BiometricConfig.quality = {"mode": "advisory", "modalities": {"fingerprint": {"min_quality": 40}}}
        verdict = assess("fingerprint", b"tmpl", Extracted(template=b"tmpl"))
        self.assertEqual(verdict.status, NOT_ASSESSED)

    def test_lower_face_uniformity_never_refuses_by_default(self):
        flat = np.tile(np.array([180, 120, 90], dtype=np.uint8), (64, 64, 1))
        rng = np.random.default_rng(5)
        flat[:32] = rng.integers(0, 256, size=(32, 64, 3), dtype=np.uint8)
        extracted = Extracted(vector=[1.0], face=FaceGeometry(box=(0, 0, 64, 64)))
        verdict = assess("face", png(flat), extracted, mode_value="advisory")
        lower = _measure(verdict.measures, "lower_face_uniformity")
        self.assertIsNotNone(lower.value)
        self.assertIsNone(lower.passed)
        self.assertEqual(verdict.status, ACCEPTED)

    def test_reasons_follow_measure_order(self):
        extracted = Extracted(vector=[1.0], quality=10.0, face=FaceGeometry(pose={"yaw": 40.0, "roll": 30.0}))
        BiometricConfig.quality = {"mode": "advisory", "modalities": {"face": {"min_quality": 50}}}
        verdict = assess("face", blurry_png(), extracted)
        self.assertEqual(
            verdict.reasons,
            ["sharpness_below_min", "yaw_above_max", "roll_above_max", "provider_quality_below_min"],
        )

    def test_mode_value_outside_modes_raises(self):
        with self.assertRaises(ImproperlyConfigured):
            assess("face", blurry_png(), Extracted(vector=[1.0]), mode_value="strict")


@skipUnless(pillow_available(), "Pillow is not importable")
class TestVerdictSerialisation(SimpleTestCase):

    def test_as_dict_is_json_safe_and_carries_no_geometry(self):
        face = FaceGeometry(
            box=(3.25, 4.75, 57.125, 58.625),
            landmarks={
                "left_eye": (21.375, 23.625), "right_eye": (41.875, 24.125), "nose": (31.625, 37.25),
                "mouth_left": (24.375, 47.875), "mouth_right": (39.625, 48.125),
            },
        )
        verdict = assess("face", sharp_png(), Extracted(vector=[1.0], face=face), mode_value="advisory")
        payload = verdict.as_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)

        coordinates = {c for point in face.landmarks.values() for c in point} | set(face.box)
        seen_keys, seen_values = set(), []

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    seen_keys.add(key)
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)
            else:
                seen_values.append(node)

        walk(payload)
        self.assertFalse({"box", "landmarks", "face", "pose"} & seen_keys)
        self.assertFalse([v for v in seen_values if isinstance(v, float) and v in coordinates])

    def test_refused_error_carries_extensions_and_reasons_only(self):
        verdict = assess("face", blurry_png(), Extracted(vector=[1.0]), mode_value="enforce")
        error = QualityRefusedError(verdict)
        self.assertIsInstance(error, ValueError)
        self.assertIs(error.verdict, verdict)
        self.assertEqual(error.extensions["code"], "BIOMETRIC_QUALITY_REFUSED")
        self.assertEqual(error.extensions["verdict"], verdict.as_dict())
        self.assertEqual(str(error), "Biometric sample refused by the quality gate: sharpness_below_min")


class TestQualityConfig(_QualityConfigTestCase):

    def test_settings_key_and_class_attribute(self):
        self.assertEqual(biometric_apps._SETTINGS_KEY_MAP["QUALITY"], "quality")
        self.assertTrue(hasattr(BiometricConfig, "quality"))
        self.assertEqual(biometric_apps.DEFAULT_CFG["quality"]["mode"], "advisory")
        self.assertEqual(biometric_apps.DEFAULT_CFG["quality"]["modalities"]["face"], FACE_DEFAULTS)

    def test_mode_only_config_keeps_builtin_face_thresholds(self):
        BiometricConfig.quality = {"mode": "enforce"}
        self.assertEqual(thresholds_for("face"), FACE_DEFAULTS)

    def test_partial_override_changes_one_key(self):
        BiometricConfig.quality = {"modalities": {"face": {"min_sharpness": 50}}}
        self.assertEqual(thresholds_for("face"), {**FACE_DEFAULTS, "min_sharpness": 50})

    def test_other_modality_defaults(self):
        self.assertEqual(thresholds_for("iris"), {"min_quality": None})

    def test_mode_rejects_unknown_value(self):
        BiometricConfig.quality = {"mode": "strict"}
        with self.assertRaises(ImproperlyConfigured):
            mode()

    def test_default_mode_is_advisory(self):
        BiometricConfig.quality = {}
        self.assertEqual(mode(), "advisory")

    def test_non_numeric_threshold_raises(self):
        BiometricConfig.quality = {"modalities": {"fingerprint": {"min_quality": "high"}}}
        with self.assertRaises(ImproperlyConfigured):
            assess("fingerprint", b"t", Extracted(template=b"t", quality=1.0), mode_value="advisory")

    def test_startup_check_logs_invalid_mode_and_missing_pillow(self):
        BiometricConfig.quality = {"mode": "strict"}
        with self.assertLogs("biometric.apps", level="ERROR"):
            BiometricConfig._check_quality_config()

        BiometricConfig.quality = {"mode": "enforce"}
        with patch("biometric.quality._pil", return_value=None), \
                self.assertLogs("biometric.quality", level="WARNING") as logs:
            BiometricConfig._check_quality_config()
        self.assertEqual(len(logs.records), 1)

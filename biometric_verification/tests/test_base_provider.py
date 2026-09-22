"""
Unit tests for providers/base.py

Covers:
- VerificationResult dataclass defaults
- _cosine_distance edge cases
- BaseBiometricProvider.verify_from_embedding default implementation
"""
import math
from django.test import SimpleTestCase
from unittest.mock import patch

from biometric_verification.providers.base import (
    BaseBiometricProvider,
    VerificationResult,
    _cosine_distance,
)


# ---------------------------------------------------------------------------
# Minimal concrete provider used only in tests
# ---------------------------------------------------------------------------

class _StubProvider(BaseBiometricProvider):
    provider_name = "stub"

    def __init__(self, embedding=None):
        self._embedding = embedding or [1.0, 0.0, 0.0]

    def verify(self, probe_image, reference_image, threshold=None):
        return VerificationResult(verified=True, provider=self.provider_name)

    def get_embedding(self, image):
        return self._embedding


# ---------------------------------------------------------------------------
# VerificationResult
# ---------------------------------------------------------------------------

class TestVerificationResult(SimpleTestCase):

    def test_defaults(self):
        r = VerificationResult(verified=True)
        self.assertTrue(r.verified)
        self.assertIsNone(r.confidence)
        self.assertIsNone(r.distance)
        self.assertIsNone(r.provider)
        self.assertIsNone(r.error)
        self.assertEqual(r.metadata, {})

    def test_all_fields(self):
        r = VerificationResult(
            verified=False,
            confidence=42.5,
            distance=0.3,
            provider="deepface",
            metadata={"model": "ArcFace"},
            error="oops",
        )
        self.assertFalse(r.verified)
        self.assertAlmostEqual(r.confidence, 42.5)
        self.assertEqual(r.provider, "deepface")
        self.assertEqual(r.metadata["model"], "ArcFace")


# ---------------------------------------------------------------------------
# _cosine_distance
# ---------------------------------------------------------------------------

class TestCosineDistance(SimpleTestCase):

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(_cosine_distance(v, v), 0.0, places=6)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(_cosine_distance([1, 0, 0], [0, 1, 0]), 1.0, places=6)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(_cosine_distance([1, 0], [-1, 0]), 2.0, places=6)

    def test_zero_vector_returns_one(self):
        self.assertEqual(_cosine_distance([0, 0, 0], [1, 2, 3]), 1.0)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _cosine_distance([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_known_value(self):
        # [1,1] vs [1,0] → cos(45°) = 1 - 1/√2 ≈ 0.2929
        result = _cosine_distance([1.0, 1.0], [1.0, 0.0])
        expected = 1.0 - 1.0 / math.sqrt(2)
        self.assertAlmostEqual(result, expected, places=6)


# ---------------------------------------------------------------------------
# BaseBiometricProvider.verify_from_embedding (default implementation)
# ---------------------------------------------------------------------------

class TestVerifyFromEmbedding(SimpleTestCase):

    def _make_provider(self, probe_embedding):
        return _StubProvider(embedding=probe_embedding)

    @patch(
        "biometric_verification.apps.BiometricVerificationConfig"
    )
    def test_identical_embeddings_verified(self, mock_cfg):
        mock_cfg.similarity_threshold = 0.68  # distance threshold = 1 - 0.68 = 0.32
        provider = self._make_provider([1.0, 0.0, 0.0])
        stored = [1.0, 0.0, 0.0]

        result = provider.verify_from_embedding(b"probe", stored)

        self.assertTrue(result.verified)
        self.assertAlmostEqual(result.distance, 0.0, places=5)
        self.assertAlmostEqual(result.confidence, 100.0, places=1)
        self.assertEqual(result.provider, "stub")

    @patch(
        "biometric_verification.apps.BiometricVerificationConfig"
    )
    def test_orthogonal_embeddings_not_verified(self, mock_cfg):
        mock_cfg.similarity_threshold = 0.68  # threshold = 0.32; distance = 1.0 > 0.32
        provider = self._make_provider([1.0, 0.0])
        stored = [0.0, 1.0]

        result = provider.verify_from_embedding(b"probe", stored)

        self.assertFalse(result.verified)
        self.assertAlmostEqual(result.distance, 1.0, places=5)

    @patch(
        "biometric_verification.apps.BiometricVerificationConfig"
    )
    def test_explicit_threshold_override(self, mock_cfg):
        mock_cfg.similarity_threshold = 0.68
        provider = self._make_provider([1.0, 0.0])
        stored = [1.0, 0.0]

        # threshold=0.0 → any non-zero distance fails
        result = provider.verify_from_embedding(b"probe", stored, threshold=0.0)
        self.assertTrue(result.verified)  # distance is 0 ≤ 0.0

    def test_get_embedding_exception_returns_error_result(self):
        class _BrokenProvider(_StubProvider):
            def get_embedding(self, image):
                raise RuntimeError("model crashed")

        provider = _BrokenProvider()
        result = provider.verify_from_embedding(b"probe", [1.0, 0.0])

        self.assertFalse(result.verified)
        self.assertIn("model crashed", result.error)

    def test_health_check_default_true(self):
        self.assertTrue(_StubProvider().health_check())

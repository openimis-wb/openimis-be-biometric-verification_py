"""
Unit tests for the ModalityProvider interface (§3.1): Extracted, the ABC
hierarchy, DeepFaceProvider's EmbeddingProvider methods (math only — no
deepface import), device_reported.DeviceReportedMatcher, and the fake
providers used everywhere else in this test suite.
"""

from django.test import SimpleTestCase

from biometric.providers.base import (
    EmbeddingProvider,
    Extracted,
    MatcherProvider,
    ModalityProvider,
)
from biometric.providers.deepface_provider import DeepFaceProvider
from biometric.providers.device_reported import DeviceReportedMatcher
from biometric.providers.fake import FakeEmbeddingProvider, FakeMatcherProvider


class TestExtracted(SimpleTestCase):

    def test_defaults(self):
        e = Extracted()
        self.assertIsNone(e.vector)
        self.assertIsNone(e.template)
        self.assertIsNone(e.template_iso)
        self.assertIsNone(e.quality)
        self.assertEqual(e.metadata, {})


class TestModalityProviderABC(SimpleTestCase):

    def test_cannot_instantiate_modality_provider_directly(self):
        with self.assertRaises(TypeError):
            ModalityProvider()

    def test_cannot_instantiate_embedding_provider_without_distance(self):
        class Incomplete(EmbeddingProvider):
            def extract(self, sample, position=None):
                return Extracted()

        with self.assertRaises(TypeError):
            Incomplete()

    def test_cannot_instantiate_matcher_provider_without_match(self):
        class Incomplete(MatcherProvider):
            def extract(self, sample, position=None):
                return Extracted()

        with self.assertRaises(TypeError):
            Incomplete()


class TestDeepFaceProviderEmbeddingInterface(SimpleTestCase):
    """Covers the EmbeddingProvider surface without importing deepface."""

    def test_identity(self):
        provider = DeepFaceProvider()
        self.assertEqual(provider.modality, "face")
        self.assertEqual(provider.kind, "embedding")
        self.assertEqual(provider.provider_name, "deepface")

    def test_default_threshold_is_fixed_similarity_value(self):
        # §6.1: the similarity-scale default (0.32), not derived from any
        # legacy config — biometric never imports biometric_verification.
        provider = DeepFaceProvider()
        self.assertAlmostEqual(provider.default_threshold, 0.32, places=6)

    def test_default_threshold_configurable_per_instance(self):
        provider = DeepFaceProvider(default_threshold=0.5)
        self.assertAlmostEqual(provider.default_threshold, 0.5, places=6)

    def test_distance_matches_legacy_cosine_distance(self):
        provider = DeepFaceProvider()
        self.assertAlmostEqual(provider.distance([1.0, 0.0], [1.0, 0.0]), 0.0, places=6)
        self.assertAlmostEqual(provider.distance([1.0, 0.0], [0.0, 1.0]), 1.0, places=6)

    def test_similarity_is_inverse_of_distance(self):
        provider = DeepFaceProvider()
        a, b = [1.0, 0.0], [0.0, 1.0]
        self.assertAlmostEqual(provider.similarity(a, b), 1.0 - provider.distance(a, b), places=6)


class TestDeviceReportedMatcher(SimpleTestCase):

    def test_extract_stores_bytes_as_given(self):
        provider = DeviceReportedMatcher(modality="fingerprint")
        extracted = provider.extract(b"vendor-template")
        self.assertEqual(extracted.template, b"vendor-template")
        self.assertIsNone(extracted.vector)

    def test_match_raises_not_implemented(self):
        provider = DeviceReportedMatcher(modality="fingerprint")
        with self.assertRaises(NotImplementedError):
            provider.match(b"a", b"b")

    def test_default_threshold_configurable(self):
        provider = DeviceReportedMatcher(modality="fingerprint", default_threshold=48)
        self.assertEqual(provider.default_threshold, 48)
        self.assertEqual(provider.provider_name, "device_reported")


class TestFakeEmbeddingProvider(SimpleTestCase):

    def test_deterministic_extraction(self):
        provider = FakeEmbeddingProvider()
        v1 = provider.extract(b"same-bytes").vector
        v2 = provider.extract(b"same-bytes").vector
        self.assertEqual(v1, v2)

    def test_different_input_different_vector(self):
        provider = FakeEmbeddingProvider()
        v1 = provider.extract(b"aaa").vector
        v2 = provider.extract(b"bbb").vector
        self.assertNotEqual(v1, v2)

    def test_identical_vectors_zero_distance(self):
        provider = FakeEmbeddingProvider()
        v = provider.extract(b"probe").vector
        self.assertAlmostEqual(provider.distance(v, v), 0.0, places=6)


class TestFakeMatcherProvider(SimpleTestCase):

    def test_match_equal_is_100(self):
        provider = FakeMatcherProvider()
        self.assertEqual(provider.match(b"same", b"same"), 100.0)

    def test_match_different_is_0(self):
        provider = FakeMatcherProvider()
        self.assertEqual(provider.match(b"a", b"b"), 0.0)

    def test_extract_wraps_bytes(self):
        provider = FakeMatcherProvider()
        self.assertEqual(provider.extract(b"tmpl").template, b"tmpl")

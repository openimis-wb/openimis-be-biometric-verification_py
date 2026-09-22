"""
Unit tests for ProviderRegistry.register_modality()/get_provider() (§3.2, §6.1).
"""

from django.test import SimpleTestCase

from biometric.apps import BiometricConfig
from biometric.providers.base import EmbeddingProvider, Extracted, MatcherProvider
from biometric.registry import ProviderRegistry


class _AlphaEmbedding(EmbeddingProvider):
    def __init__(self, modality="face", default_threshold=0.5, **kwargs):
        self.modality = modality
        self.provider_name = "alpha"
        self.default_threshold = default_threshold
        self.extra = kwargs.get("extra")

    def extract(self, sample, position=None):
        return Extracted(vector=[1.0])

    def distance(self, a, b):
        return 0.0


class _BetaMatcher(MatcherProvider):
    def __init__(self, modality="fingerprint", default_threshold=50, **kwargs):
        self.modality = modality
        self.provider_name = "beta"
        self.default_threshold = default_threshold

    def extract(self, sample, position=None):
        return Extracted(template=sample)

    def match(self, probe, reference):
        return 100.0


class TestRegisterModality(SimpleTestCase):

    def setUp(self):
        self._registry = dict(ProviderRegistry._modality_registry)
        self._instances = dict(ProviderRegistry._modality_instances)
        self._modalities = BiometricConfig.modalities
        ProviderRegistry._modality_registry.clear()
        ProviderRegistry._modality_instances.clear()

    def tearDown(self):
        ProviderRegistry._modality_registry.clear()
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry._modality_registry.update(self._registry)
        ProviderRegistry._modality_instances.update(self._instances)
        BiometricConfig.modalities = self._modalities

    def test_register_adds_to_modality_registry(self):
        ProviderRegistry.register_modality("face", "alpha", _AlphaEmbedding)
        self.assertIn(("face", "alpha"), ProviderRegistry._modality_registry)

    def test_register_non_modality_provider_raises_type_error(self):
        with self.assertRaises(TypeError):
            ProviderRegistry.register_modality("face", "bad", object)

    def test_get_provider_instantiates_configured_provider(self):
        BiometricConfig.modalities = {"face": {"provider": "alpha", "threshold": 0.42}}
        ProviderRegistry.register_modality("face", "alpha", _AlphaEmbedding)

        provider = ProviderRegistry.get_provider("face")

        self.assertIsInstance(provider, _AlphaEmbedding)
        self.assertEqual(provider.modality, "face")
        self.assertEqual(provider.default_threshold, 0.42)

    def test_get_provider_is_cached(self):
        BiometricConfig.modalities = {"face": {"provider": "alpha"}}
        ProviderRegistry.register_modality("face", "alpha", _AlphaEmbedding)

        p1 = ProviderRegistry.get_provider("face")
        p2 = ProviderRegistry.get_provider("face")

        self.assertIs(p1, p2)

    def test_get_provider_unknown_name_raises_key_error(self):
        BiometricConfig.modalities = {"face": {"provider": "nonexistent"}}
        with self.assertRaises(KeyError):
            ProviderRegistry.get_provider("face")

    def test_get_provider_missing_modality_config_raises_key_error(self):
        BiometricConfig.modalities = {}
        with self.assertRaises(KeyError):
            ProviderRegistry.get_provider("iris")

    def test_different_modalities_resolve_independently(self):
        BiometricConfig.modalities = {
            "face": {"provider": "alpha"},
            "fingerprint": {"provider": "beta", "threshold": 48},
        }
        ProviderRegistry.register_modality("face", "alpha", _AlphaEmbedding)
        ProviderRegistry.register_modality("fingerprint", "beta", _BetaMatcher)

        face = ProviderRegistry.get_provider("face")
        fingerprint = ProviderRegistry.get_provider("fingerprint")

        self.assertIsInstance(face, _AlphaEmbedding)
        self.assertIsInstance(fingerprint, _BetaMatcher)
        self.assertEqual(fingerprint.default_threshold, 48)


class TestBuiltinModalityRegistrations(SimpleTestCase):
    """The registry module registers built-ins at import time — verify they landed."""

    def test_device_reported_registered_for_documented_modalities(self):
        for modality in ("face", "fingerprint", "voice", "iris", "palmvein"):
            self.assertIn((modality, "device_reported"), ProviderRegistry._modality_registry)

    def test_deepface_registered_for_face_when_importable(self):
        # deepface itself is not installed in the test environment; the face/deepface
        # key is only present when the guarded import in registry.py succeeded.
        try:
            import deepface  # noqa: F401
        except ImportError:
            self.skipTest("deepface not installed")
        self.assertIn(("face", "deepface"), ProviderRegistry._modality_registry)

"""
Unit tests for ProviderRegistry.register_modality()/get_provider() (§3.2).

Uses a separate dict from the legacy register()/get_active_provider() —
these tests clear/restore only the modality-keyed state so test_registry.py
is unaffected.
"""

from django.test import SimpleTestCase

from biometric_verification.apps import BiometricVerificationConfig
from biometric_verification.providers.base import (
    BaseBiometricProvider,
    EmbeddingProvider,
    Extracted,
    MatcherProvider,
    VerificationResult,
)
from biometric_verification.registry import ProviderRegistry


class _LegacyStubProvider(BaseBiometricProvider):
    provider_name = "legacy-stub"

    def verify(self, probe_image, reference_image, threshold=None):
        return VerificationResult(verified=True, provider=self.provider_name)

    def get_embedding(self, image):
        return [0.1, 0.2]


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
        self._modalities = BiometricVerificationConfig.modalities
        ProviderRegistry._modality_registry.clear()
        ProviderRegistry._modality_instances.clear()

    def tearDown(self):
        ProviderRegistry._modality_registry.clear()
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry._modality_registry.update(self._registry)
        ProviderRegistry._modality_instances.update(self._instances)
        BiometricVerificationConfig.modalities = self._modalities

    def test_register_adds_to_modality_registry(self):
        ProviderRegistry.register_modality("face", "alpha", _AlphaEmbedding)
        self.assertIn(("face", "alpha"), ProviderRegistry._modality_registry)

    def test_register_non_modality_provider_raises_type_error(self):
        with self.assertRaises(TypeError):
            ProviderRegistry.register_modality("face", "bad", object)

    def test_does_not_touch_legacy_registry(self):
        ProviderRegistry.register_modality("face", "alpha", _AlphaEmbedding)
        self.assertNotIn("alpha", ProviderRegistry._registry)

    def test_get_provider_instantiates_configured_provider(self):
        BiometricVerificationConfig.modalities = {"face": {"provider": "alpha", "threshold": 0.42}}
        ProviderRegistry.register_modality("face", "alpha", _AlphaEmbedding)

        provider = ProviderRegistry.get_provider("face")

        self.assertIsInstance(provider, _AlphaEmbedding)
        self.assertEqual(provider.modality, "face")
        self.assertEqual(provider.default_threshold, 0.42)

    def test_get_provider_is_cached(self):
        BiometricVerificationConfig.modalities = {"face": {"provider": "alpha"}}
        ProviderRegistry.register_modality("face", "alpha", _AlphaEmbedding)

        p1 = ProviderRegistry.get_provider("face")
        p2 = ProviderRegistry.get_provider("face")

        self.assertIs(p1, p2)

    def test_get_provider_unknown_name_raises_key_error(self):
        BiometricVerificationConfig.modalities = {"face": {"provider": "nonexistent"}}
        with self.assertRaises(KeyError):
            ProviderRegistry.get_provider("face")

    def test_get_provider_missing_modality_config_raises_key_error(self):
        BiometricVerificationConfig.modalities = {}
        with self.assertRaises(KeyError):
            ProviderRegistry.get_provider("iris")

    def test_different_modalities_resolve_independently(self):
        BiometricVerificationConfig.modalities = {
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


class TestLegacyGetActiveProviderUnchanged(SimpleTestCase):
    """
    §3.7: "legacy get_active_provider unchanged". The pre-existing
    test_registry.py exercises this via @patch("...registry.BiometricVerificationConfig"),
    which fails because get_active_provider() imports it locally (unrelated to
    this branch's changes — see the final report). This test proves the same
    behaviour by setting the real config class attribute directly instead.
    """

    def setUp(self):
        self._registry = dict(ProviderRegistry._registry)
        self._instances = dict(ProviderRegistry._instances)
        self._provider = BiometricVerificationConfig.provider
        self._provider_config = BiometricVerificationConfig.provider_config

    def tearDown(self):
        ProviderRegistry._registry.clear()
        ProviderRegistry._instances.clear()
        ProviderRegistry._registry.update(self._registry)
        ProviderRegistry._instances.update(self._instances)
        BiometricVerificationConfig.provider = self._provider
        BiometricVerificationConfig.provider_config = self._provider_config

    def test_returns_cached_instance_for_configured_provider(self):
        ProviderRegistry.register("legacy-stub", _LegacyStubProvider)
        BiometricVerificationConfig.provider = "legacy-stub"
        BiometricVerificationConfig.provider_config = {}

        p1 = ProviderRegistry.get_active_provider()
        p2 = ProviderRegistry.get_active_provider()

        self.assertIsInstance(p1, _LegacyStubProvider)
        self.assertIs(p1, p2)

    def test_unknown_provider_still_raises_key_error(self):
        BiometricVerificationConfig.provider = "nonexistent-legacy-provider"
        with self.assertRaises(KeyError):
            ProviderRegistry.get_active_provider()

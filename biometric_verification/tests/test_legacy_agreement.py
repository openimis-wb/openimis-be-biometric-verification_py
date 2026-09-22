"""
Regression test: legacy verify_from_embedding() (biometric_verification) and
new verify() (biometric) must reach the same verdict for the same pair, at
the default config, on the real DeepFace threshold/decision math.

Only collected where biometric_verification is installed as a Django app —
which (since it was restored to exact upstream bytes, §6.1) requires insuree
and claim, so this belongs to the health test environment (§6.4), not the
generic one. See docs/wb-biometric-dedup-seam.md §6.4.
"""

from django.apps import apps
from django.test import TestCase

if not apps.is_installed("biometric_verification"):
    import pytest
    pytest.skip(
        "requires biometric_verification installed (health environment)",
        allow_module_level=True,
    )


class TestLegacyAndNewVerifyAgreeOnDeepFaceThreshold(TestCase):
    """
    60 degrees apart, both unit vectors: cosine distance = 0.5, similarity = 0.5.
    Legacy: 0.5 <= 0.68 -> verified. new path: 0.5 >= 0.32 -> verified. Same verdict.
    """

    PROBE_VECTOR = [1.0, 0.0]
    REFERENCE_VECTOR = [0.5, 0.8660254037844387]

    def setUp(self):
        from biometric.apps import BiometricConfig
        from biometric.providers.deepface_provider import DeepFaceProvider as NewDeepFaceProvider
        from biometric.registry import ProviderRegistry
        from biometric_verification.apps import BiometricVerificationConfig
        from biometric_verification.providers.deepface_provider import DeepFaceProvider as LegacyDeepFaceProvider

        self._NewDeepFaceProvider = NewDeepFaceProvider
        self._LegacyDeepFaceProvider = LegacyDeepFaceProvider
        self._BiometricConfig = BiometricConfig
        self._BiometricVerificationConfig = BiometricVerificationConfig

        self._legacy_threshold = BiometricVerificationConfig.similarity_threshold
        self._modalities = BiometricConfig.modalities
        self._registry_snapshot = dict(ProviderRegistry._modality_registry)
        self._instances_snapshot = dict(ProviderRegistry._modality_instances)

        BiometricVerificationConfig.similarity_threshold = 0.68
        BiometricConfig.modalities = {"face": {"provider": "deepface", "threshold": 0.32}}
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry.register_modality("face", "deepface", NewDeepFaceProvider)

        from unittest.mock import patch
        self._get_embedding_patch = patch.object(
            LegacyDeepFaceProvider, "get_embedding", return_value=self.PROBE_VECTOR,
        )
        self._get_embedding_patch.start()

    def tearDown(self):
        from biometric.registry import ProviderRegistry

        self._get_embedding_patch.stop()
        self._BiometricVerificationConfig.similarity_threshold = self._legacy_threshold
        self._BiometricConfig.modalities = self._modalities
        ProviderRegistry._modality_registry.clear()
        ProviderRegistry._modality_registry.update(self._registry_snapshot)
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry._modality_instances.update(self._instances_snapshot)

    def test_same_verdict_at_default_thresholds(self):
        from biometric.models import BiometricTemplate
        from biometric.services import verify

        legacy_provider = self._LegacyDeepFaceProvider()
        legacy_result = legacy_provider.verify_from_embedding(
            probe_image=b"unused-because-get_embedding-is-stubbed",
            reference_embedding=self.REFERENCE_VECTOR,
        )

        BiometricTemplate.objects.create(
            subject_model="individual.Individual", subject_id="s1", modality="face", kind="embedding",
            vector=self.REFERENCE_VECTOR, provider="deepface", model_name=legacy_provider.model_name,
        )
        new_result = verify(
            "individual.Individual", "s1", "face", sample=b"unused", actor="tester",
        )

        self.assertTrue(legacy_result.verified, "legacy path should verify this pair")
        self.assertEqual(legacy_result.verified, new_result.verified)

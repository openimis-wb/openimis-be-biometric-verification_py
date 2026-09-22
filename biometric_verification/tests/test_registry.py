"""
Unit tests for registry.py

Covers:
- Registering a provider
- get_active_provider() returns correct instance
- Provider instance is cached (singleton per name)
- Unknown provider raises KeyError
- Invalid class raises TypeError
- available_providers() reflects registrations
"""
from django.test import SimpleTestCase
from unittest.mock import patch

from biometric_verification.providers.base import BaseBiometricProvider, VerificationResult
from biometric_verification.registry import ProviderRegistry


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _AlphaProvider(BaseBiometricProvider):
    provider_name = "alpha"

    def verify(self, probe_image, reference_image, threshold=None):
        return VerificationResult(verified=True, provider=self.provider_name)

    def get_embedding(self, image):
        return [0.1, 0.2]


class _BetaProvider(BaseBiometricProvider):
    provider_name = "beta"

    def __init__(self, model_name="BetaNet", **kwargs):
        self.model_name = model_name

    def verify(self, probe_image, reference_image, threshold=None):
        return VerificationResult(verified=False, provider=self.provider_name)

    def get_embedding(self, image):
        return [0.3, 0.4]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProviderRegistry(SimpleTestCase):

    def setUp(self):
        # Isolate each test from the global registry state.
        self._original_registry = dict(ProviderRegistry._registry)
        self._original_instances = dict(ProviderRegistry._instances)
        ProviderRegistry._registry.clear()
        ProviderRegistry._instances.clear()

    def tearDown(self):
        ProviderRegistry._registry.clear()
        ProviderRegistry._instances.clear()
        ProviderRegistry._registry.update(self._original_registry)
        ProviderRegistry._instances.update(self._original_instances)

    # --- register() ---

    def test_register_adds_to_registry(self):
        ProviderRegistry.register("alpha", _AlphaProvider)
        self.assertIn("alpha", ProviderRegistry._registry)

    def test_register_non_provider_raises_type_error(self):
        with self.assertRaises(TypeError):
            ProviderRegistry.register("bad", object)

    def test_re_register_invalidates_cached_instance(self):
        ProviderRegistry.register("alpha", _AlphaProvider)
        # Force an instance into cache
        ProviderRegistry._instances["alpha"] = _AlphaProvider()
        # Re-register should evict it
        ProviderRegistry.register("alpha", _AlphaProvider)
        self.assertNotIn("alpha", ProviderRegistry._instances)

    # --- get_active_provider() ---

    @patch("biometric_verification.apps.BiometricVerificationConfig")
    def test_returns_correct_provider_instance(self, mock_cfg):
        mock_cfg.provider = "alpha"
        mock_cfg.provider_config = {}
        ProviderRegistry.register("alpha", _AlphaProvider)

        provider = ProviderRegistry.get_active_provider()

        self.assertIsInstance(provider, _AlphaProvider)

    @patch("biometric_verification.apps.BiometricVerificationConfig")
    def test_instance_is_cached(self, mock_cfg):
        mock_cfg.provider = "alpha"
        mock_cfg.provider_config = {}
        ProviderRegistry.register("alpha", _AlphaProvider)

        p1 = ProviderRegistry.get_active_provider()
        p2 = ProviderRegistry.get_active_provider()

        self.assertIs(p1, p2)

    @patch("biometric_verification.apps.BiometricVerificationConfig")
    def test_provider_config_passed_as_kwargs(self, mock_cfg):
        mock_cfg.provider = "beta"
        mock_cfg.provider_config = {"model_name": "CustomNet"}
        ProviderRegistry.register("beta", _BetaProvider)

        provider = ProviderRegistry.get_active_provider()

        self.assertEqual(provider.model_name, "CustomNet")

    @patch("biometric_verification.apps.BiometricVerificationConfig")
    def test_unknown_provider_raises_key_error(self, mock_cfg):
        mock_cfg.provider = "nonexistent"
        mock_cfg.provider_config = {}

        with self.assertRaises(KeyError) as ctx:
            ProviderRegistry.get_active_provider()

        self.assertIn("nonexistent", str(ctx.exception))

    # --- available_providers() ---

    def test_available_providers_lists_registered(self):
        ProviderRegistry.register("alpha", _AlphaProvider)
        ProviderRegistry.register("beta", _BetaProvider)

        available = ProviderRegistry.available_providers()

        self.assertIn("alpha", available)
        self.assertIn("beta", available)

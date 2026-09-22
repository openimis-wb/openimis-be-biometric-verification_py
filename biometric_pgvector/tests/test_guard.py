"""
Plaintext guard (docs/wb-biometric-dedup-seam.md §6.2): biometric_pgvector
refuses to start when templates are encrypted at rest unless plaintext
indexing is explicitly allowed. Exercised directly against the guard, not
by re-triggering the real app-ready (which only runs once per process).
"""

from unittest.mock import patch

from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase

from biometric.apps import BiometricConfig
from biometric_pgvector.apps import BiometricPgvectorConfig

RAW_CFG = "biometric_pgvector.config._raw_biometric_cfg"


class TestPlaintextGuard(SimpleTestCase):

    def test_raises_when_template_key_set_and_plaintext_not_allowed(self):
        with patch(RAW_CFG, return_value={"template_key": "a-fernet-key"}):
            with self.assertRaises(ImproperlyConfigured):
                BiometricPgvectorConfig._guard_plaintext_index()

    def test_passes_when_template_key_set_and_plaintext_allowed(self):
        cfg = {"template_key": "a-fernet-key", "allow_plaintext_index": True}
        with patch(RAW_CFG, return_value=cfg):
            BiometricPgvectorConfig._guard_plaintext_index()

    def test_passes_when_no_template_key(self):
        with patch(RAW_CFG, return_value={}):
            BiometricPgvectorConfig._guard_plaintext_index()

    def test_raises_before_biometric_has_loaded_its_config(self):
        """biometric_pgvector's ready() may run first when a manifest lists it
        before biometric: BiometricConfig.template_key is then still None, and
        the guard must still see the key in the configuration."""
        with patch.object(BiometricConfig, "template_key", None), \
                patch(RAW_CFG, return_value={"template_key": "a-fernet-key"}):
            with self.assertRaises(ImproperlyConfigured):
                BiometricPgvectorConfig._guard_plaintext_index()

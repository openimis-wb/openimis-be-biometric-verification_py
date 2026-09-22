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


class TestPlaintextGuard(SimpleTestCase):

    def setUp(self):
        super().setUp()
        self._template_key = BiometricConfig.template_key

    def tearDown(self):
        super().tearDown()
        BiometricConfig.template_key = self._template_key

    def test_raises_when_template_key_set_and_plaintext_not_allowed(self):
        BiometricConfig.template_key = "a-fernet-key"
        with patch("biometric_pgvector.config.allow_plaintext_index", return_value=False):
            with self.assertRaises(ImproperlyConfigured):
                BiometricPgvectorConfig._guard_plaintext_index()

    def test_passes_when_template_key_set_and_plaintext_allowed(self):
        BiometricConfig.template_key = "a-fernet-key"
        with patch("biometric_pgvector.config.allow_plaintext_index", return_value=True):
            BiometricPgvectorConfig._guard_plaintext_index()

    def test_passes_when_no_template_key(self):
        BiometricConfig.template_key = None
        with patch("biometric_pgvector.config.allow_plaintext_index", return_value=False):
            BiometricPgvectorConfig._guard_plaintext_index()

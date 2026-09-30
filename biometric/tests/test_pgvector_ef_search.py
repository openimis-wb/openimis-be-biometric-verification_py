"""
biometric_pgvector reads HNSW_EF_SEARCH once, in BiometricPgvectorConfig.ready(),
from the same source and override order as its other keys. The pgvector
identify path uses that value and never re-reads the configuration.
Runs without the vector extension: the database cursor is a double.
"""

import contextlib
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from biometric import services
from biometric.providers.fake import FakeEmbeddingProvider
from biometric_pgvector.apps import BiometricPgvectorConfig


class TestEfSearchReadOnce(SimpleTestCase):

    def setUp(self):
        self.addCleanup(setattr, BiometricPgvectorConfig, "hnsw_ef_search", BiometricPgvectorConfig.hnsw_ef_search)

    def test_default_is_200(self):
        with patch("biometric_pgvector.config._raw_biometric_cfg", return_value={}):
            BiometricPgvectorConfig._load_settings()
        self.assertEqual(BiometricPgvectorConfig.hnsw_ef_search, 200)

    @override_settings(BIOMETRIC={"HNSW_EF_SEARCH": 64})
    def test_settings_override(self):
        with patch("core.models.ModuleConfiguration.get_or_default", return_value={"hnsw_ef_search": 400}):
            BiometricPgvectorConfig._load_settings()
        self.assertEqual(BiometricPgvectorConfig.hnsw_ef_search, 64)

    def test_identify_uses_the_loaded_value_without_reading_the_configuration(self):
        BiometricPgvectorConfig.hnsw_ef_search = 77
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor

        with patch("django.apps.apps.is_installed", return_value=True), \
                patch("django.db.connection", connection), \
                patch("django.db.transaction.atomic", contextlib.nullcontext), \
                patch("biometric_pgvector.config._raw_biometric_cfg", side_effect=AssertionError("re-read")), \
                patch("core.models.ModuleConfiguration.get_or_default", side_effect=AssertionError("re-read")):
            for _ in range(3):
                services._identify_pgvector(FakeEmbeddingProvider(), "face", [1.0, 0.0], 5, None, None)

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertEqual(statements.count("SET LOCAL hnsw.ef_search = 77"), 3)

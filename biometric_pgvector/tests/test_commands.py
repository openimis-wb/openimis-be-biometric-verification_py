"""biometric_vector_index management command (docs/wb-biometric-dedup-seam.md §6.2)."""

from django.core.management import call_command
from django.db import connection
from django.test import TestCase

from biometric.services import enrol
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase
from biometric_pgvector.management.commands.biometric_vector_index import index_name


def _index_exists(name):
    with connection.cursor() as cursor:
        cursor.execute("SELECT indexname FROM pg_indexes WHERE indexname = %s", [name])
        return cursor.fetchone() is not None


class TestVectorIndexCommand(_MultimodalServiceTestCase):

    def test_creates_the_named_partial_hnsw_index(self):
        enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="tester")
        # TestCase wraps the test in one uncommitted transaction: the row's
        # deferred FK trigger is still pending, and CREATE INDEX refuses to
        # run against a table with pending trigger events. Force it resolved
        # now, exactly as a real COMMIT would before a standalone command run.
        with connection.cursor() as cursor:
            cursor.execute("SET CONSTRAINTS ALL IMMEDIATE")

        name = index_name("fake_embedding_model", 8)
        self.assertFalse(_index_exists(name))

        call_command("biometric_vector_index", "--model", "fake_embedding_model", "--dim", "8")

        self.assertTrue(_index_exists(name))

        with connection.cursor() as cursor:
            cursor.execute("SELECT indexdef FROM pg_indexes WHERE indexname = %s", [name])
            indexdef = cursor.fetchone()[0]
        self.assertIn("hnsw", indexdef)
        self.assertIn("vector_cosine_ops", indexdef)
        self.assertIn("model_name", indexdef)

    def test_is_idempotent(self):
        call_command("biometric_vector_index", "--model", "fake_embedding_model", "--dim", "8")
        call_command("biometric_vector_index", "--model", "fake_embedding_model", "--dim", "8")
        self.assertTrue(_index_exists(index_name("fake_embedding_model", 8)))

    def test_drop_removes_the_index(self):
        call_command("biometric_vector_index", "--model", "fake_embedding_model", "--dim", "8")
        name = index_name("fake_embedding_model", 8)
        self.assertTrue(_index_exists(name))

        call_command("biometric_vector_index", "--model", "fake_embedding_model", "--dim", "8", "--drop")

        self.assertFalse(_index_exists(name))

    def test_different_dimensions_get_different_indexes(self):
        call_command("biometric_vector_index", "--model", "m", "--dim", "8")
        call_command("biometric_vector_index", "--model", "m", "--dim", "128")

        self.assertTrue(_index_exists(index_name("m", 8)))
        self.assertTrue(_index_exists(index_name("m", 128)))

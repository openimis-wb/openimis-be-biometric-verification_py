"""
A row marked encrypted that does not decrypt raises crypto.TemplateKeyError:
a wrong or rotated TEMPLATE_KEY, or no key at all. It is never read as the
stored ciphertext, never turned into verified=false, and its text never
quotes the ciphertext. Rows stored with encrypted=False are read as stored.
"""

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import graphene
from cryptography.fernet import Fernet
from django.test import SimpleTestCase

from biometric import crypto
from biometric.crypto import TemplateKeyError
from biometric.dedup_source import BiometricCandidateSource
from biometric.models import BiometricTemplate, BiometricVerification
from biometric.apps import BiometricConfig
from biometric.providers.fake import FakeEmbeddingProvider
from biometric.services import enrol, identify, templates_of, verify
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase


class TestCryptoRaises(SimpleTestCase):

    def test_vector_under_another_key_raises_without_the_ciphertext(self):
        ciphertext = crypto.encrypt_vector([1.0, 2.0], Fernet.generate_key())

        with self.assertRaises(TemplateKeyError) as raised:
            crypto.decrypt_vector(ciphertext, Fernet.generate_key())

        self.assertNotIn(ciphertext, str(raised.exception))
        self.assertEqual(raised.exception.extensions, {"code": "BIOMETRIC_TEMPLATE_KEY"})

    def test_bytes_under_another_key_raises_without_the_ciphertext(self):
        ciphertext = crypto.encrypt_bytes(b"template", Fernet.generate_key())

        with self.assertRaises(TemplateKeyError) as raised:
            crypto.decrypt_bytes(ciphertext, Fernet.generate_key())

        self.assertNotIn(ciphertext.decode(), str(raised.exception))

    def test_row_key_follows_the_encrypted_flag(self):
        key = Fernet.generate_key()
        self.assertIs(crypto.row_key(True, key), key)
        self.assertIsNone(crypto.row_key(False, key))
        self.assertIsNone(crypto.row_key(False, None))

    def test_row_key_raises_for_an_encrypted_row_without_a_key(self):
        with self.assertRaises(TemplateKeyError):
            crypto.row_key(True, None)


class _EncryptedGalleryTestCase(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.template_key = Fernet.generate_key()
        self.face = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="tester")
        self.finger = enrol(SUBJECT_MODEL, "s1", "fingerprint", b"print", actor="tester")
        self.ciphertext = self.face.vector

    def _rotate(self):
        BiometricConfig.template_key = Fernet.generate_key()

    def _drop_key(self):
        BiometricConfig.template_key = None


class TestVerifyRaises(_EncryptedGalleryTestCase):

    def test_wrong_key_raises_and_records_no_verification(self):
        self._rotate()

        with self.assertLogs("biometric.services", level="ERROR") as logs:
            with self.assertRaises(TemplateKeyError) as raised:
                verify(SUBJECT_MODEL, "s1", "face", sample=b"photo", actor="agent")

        self.assertEqual(BiometricVerification.objects.count(), 0)
        self.assertNotIn(self.ciphertext, str(raised.exception))
        self.assertIn(str(self.face.id), logs.output[0])
        self.assertNotIn(self.ciphertext, "".join(logs.output))

    def test_missing_key_on_an_encrypted_row_raises(self):
        self._drop_key()

        with self.assertRaises(TemplateKeyError):
            verify(SUBJECT_MODEL, "s1", "fingerprint", sample=b"print", actor="agent")
        self.assertEqual(BiometricVerification.objects.count(), 0)

    def test_plaintext_rows_are_read_as_stored(self):
        vector = FakeEmbeddingProvider().extract(b"photo").vector
        BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id="s2", modality="face", kind="embedding",
            vector=vector, encrypted=False, provider="fake_embedding", model_name="",
        )

        result = verify(SUBJECT_MODEL, "s2", "face", sample=b"photo", actor="agent")

        self.assertTrue(result.verified)


class TestReadPathsRaise(_EncryptedGalleryTestCase):

    def test_identify_numpy_path(self):
        self._rotate()
        with self.assertRaises(TemplateKeyError):
            identify("face", sample=b"photo")

    def test_identify_template_path(self):
        self._drop_key()
        with self.assertRaises(TemplateKeyError):
            identify("fingerprint", sample=b"print")

    def test_templates_of(self):
        self._rotate()
        with self.assertRaises(TemplateKeyError):
            templates_of(SUBJECT_MODEL, "s1", actor="reader")

    def test_candidate_scan(self):
        enrol(SUBJECT_MODEL, "s2", "face", b"photo", actor="tester")
        self._rotate()
        with self.assertRaises(TemplateKeyError):
            list(BiometricCandidateSource("face").scan(None))


class TestGraphQLError(_EncryptedGalleryTestCase):

    def _execute(self, query):
        from biometric.schema import Mutation, Query

        user = MagicMock(is_anonymous=False, username="agent")
        user.has_perms.return_value = True
        schema = graphene.Schema(query=Query, mutation=Mutation)
        return schema.execute(query, context_value=SimpleNamespace(user=user, headers={}))

    def test_verify_and_identify_report_the_code_not_the_ciphertext(self):
        self._rotate()
        sample = base64.b64encode(b"photo").decode()
        queries = {
            "verifyBiometric": 'mutation { verifyBiometric(subjectId: "s1", modality: "face", sample: "%s") '
                               "{ verified } }" % sample,
            "identifyBiometric": 'query { identifyBiometric(modality: "face", sample: "%s") { subjectId } }' % sample,
        }
        for field, query in queries.items():
            with self.subTest(field):
                result = self._execute(query)
                self.assertIsNone((result.data or {}).get(field))
                self.assertIsInstance(result.errors[0].original_error, TemplateKeyError)
                self.assertEqual(result.errors[0].extensions, {"code": "BIOMETRIC_TEMPLATE_KEY"})
                self.assertNotIn(self.ciphertext, str(result.errors[0]))
        self.assertEqual(BiometricVerification.objects.count(), 0)

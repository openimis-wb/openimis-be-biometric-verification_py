"""
BiometricCandidateSource.scan decrypts the gallery once per scan and ranks
every probe against it: N encrypted templates cost N decrypts, not N + N².
The candidates are the ones identify() yields per probe.
"""

from unittest.mock import patch

from cryptography.fernet import Fernet

from biometric import crypto
from biometric.apps import BiometricConfig
from biometric.dedup_source import BiometricCandidateSource
from biometric.services import enrol, identify
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase


class _Counter:

    def __init__(self, function):
        self.function = function
        self.calls = 0

    def __call__(self, value, key):
        if value is not None:
            self.calls += 1
        return self.function(value, key)


class TestScanDecryptsOnce(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.template_key = Fernet.generate_key()
        BiometricConfig.vector_index = "numpy"
        BiometricConfig.dedup_threshold = {"face": 0.99, "fingerprint": 50.0}
        for subject, sample in (("a", b"one"), ("b", b"one"), ("c", b"two"), ("d", b"two"), ("e", b"three")):
            enrol(SUBJECT_MODEL, subject, "face", sample, actor="agent")
            enrol(SUBJECT_MODEL, subject, "fingerprint", sample, actor="agent")

    def _scan(self, modality, function_name):
        counter = _Counter(getattr(crypto, function_name))
        with patch.object(crypto, function_name, counter):
            candidates = list(BiometricCandidateSource(modality).scan(None))
        return counter.calls, sorted((c.subject_a, c.subject_b) for c in candidates)

    def test_embedding_gallery(self):
        calls, pairs = self._scan("face", "decrypt_vector")

        self.assertEqual(calls, 5)
        self.assertEqual(pairs, [("a", "b"), ("c", "d")])

    def test_template_gallery(self):
        calls, pairs = self._scan("fingerprint", "decrypt_bytes")

        self.assertEqual(calls, 5)
        self.assertEqual(pairs, [("a", "b"), ("c", "d")])

    def test_same_matches_as_identify_per_probe(self):
        for modality, sample in (("face", b"one"), ("fingerprint", b"one")):
            with self.subTest(modality):
                probe = identify(modality, sample=sample, exclude_subject="a")
                candidates = list(BiometricCandidateSource(modality).scan(None))
                expected = {m.subject_id for m in probe if m.score >= BiometricConfig.dedup_threshold[modality]}
                found = {c.subject_b if c.subject_a == "a" else c.subject_a
                         for c in candidates if "a" in (c.subject_a, c.subject_b)}
                self.assertEqual(found, expected)

"""
Unit tests for dedup_source.BiometricCandidateSource (§3.5, §3.7): threshold,
pair ordering, and watermark advance. Exercises whichever Candidate/Watermark
shape is active in this environment (real deduplication.sources here, since
it's installed side-by-side — see dedup_source.py's try/except).
"""

from django.test import TestCase

from biometric.apps import BiometricConfig
from biometric.dedup_source import BiometricCandidateSource, Watermark, order_pair
from biometric.models import BiometricTemplate
from biometric.providers.fake import FakeEmbeddingProvider
from biometric.registry import ProviderRegistry

SUBJECT_MODEL = "individual.Individual"


class _DedupSourceTestCase(TestCase):
    """
    identify() (called by scan()) filters the gallery on the modality's
    configured provider/model_name, so the fake provider registered here must
    match the provider/model_name stamped on the manually-created rows below.
    """

    def setUp(self):
        super().setUp()
        self._dedup_threshold = BiometricConfig.dedup_threshold
        self._modalities = BiometricConfig.modalities
        BiometricConfig.dedup_threshold = {"face": 0.9}
        BiometricConfig.modalities = {"face": {"provider": "fake_embedding", "threshold": 0.68}}
        self._registry_snapshot = dict(ProviderRegistry._modality_registry)
        self._instances_snapshot = dict(ProviderRegistry._modality_instances)
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry.register_modality("face", "fake_embedding", FakeEmbeddingProvider)

    def tearDown(self):
        super().tearDown()
        BiometricConfig.dedup_threshold = self._dedup_threshold
        BiometricConfig.modalities = self._modalities
        ProviderRegistry._modality_registry.clear()
        ProviderRegistry._modality_registry.update(self._registry_snapshot)
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry._modality_instances.update(self._instances_snapshot)

    def _make_template(self, subject_id, vector):
        return BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id=subject_id, modality="face",
            kind="embedding", vector=vector, provider="fake_embedding", model_name="",
        )


class TestScan(_DedupSourceTestCase):

    def test_yields_candidate_at_or_above_threshold(self):
        self._make_template("b-subject", [1.0, 0.0])
        self._make_template("a-subject", [1.0, 0.0])  # identical -> similarity 1.0

        source = BiometricCandidateSource(modality="face")
        candidates = list(source.scan(None))

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.kind, "biometric")
        self.assertGreaterEqual(candidate.score, 0.9)

    def test_below_threshold_yields_nothing(self):
        self._make_template("a-subject", [1.0, 0.0])
        self._make_template("b-subject", [0.0, 1.0])  # orthogonal -> similarity 0.0

        source = BiometricCandidateSource(modality="face")
        candidates = list(source.scan(None))

        self.assertEqual(candidates, [])

    def test_pair_is_ordered_with_order_pair(self):
        self._make_template("zzz", [1.0, 0.0])
        self._make_template("aaa", [1.0, 0.0])

        source = BiometricCandidateSource(modality="face")
        candidates = list(source.scan(None))

        self.assertEqual(len(candidates), 1)
        expected = order_pair("zzz", "aaa")
        self.assertEqual((candidates[0].subject_a, candidates[0].subject_b), expected)
        self.assertLess(candidates[0].subject_a, candidates[0].subject_b)

    def test_evidence_names_the_two_templates(self):
        t1 = self._make_template("a-subject", [1.0, 0.0])
        t2 = self._make_template("b-subject", [1.0, 0.0])

        source = BiometricCandidateSource(modality="face")
        candidates = list(source.scan(None))

        evidence = candidates[0].evidence
        self.assertEqual(evidence["modality"], "face")
        self.assertEqual(evidence["provider"], "fake_embedding")
        self.assertEqual({evidence["template_a"], evidence["template_b"]}, {str(t1.id), str(t2.id)})

    def test_scan_ignores_other_modalities(self):
        BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id="a", modality="fingerprint",
            kind="template", template=b"x", provider="device_reported", model_name="",
        )
        source = BiometricCandidateSource(modality="face")
        self.assertEqual(list(source.scan(None)), [])


class TestWatermark(_DedupSourceTestCase):

    def test_none_when_no_active_templates(self):
        source = BiometricCandidateSource(modality="face")
        watermark = source.watermark()
        self.assertIsNone(watermark.updated_at)
        self.assertIsNone(watermark.last_id)

    def test_advances_to_newest_active_template(self):
        self._make_template("a", [1.0, 0.0])
        newest = self._make_template("b", [0.0, 1.0])

        source = BiometricCandidateSource(modality="face")
        watermark = source.watermark()

        self.assertEqual(watermark.last_id, str(newest.id))
        self.assertIsNotNone(watermark.updated_at)

    def test_scan_since_watermark_skips_already_seen_rows(self):
        self._make_template("a-subject", [1.0, 0.0])
        first_watermark = BiometricCandidateSource(modality="face").watermark()

        self._make_template("b-subject", [1.0, 0.0])  # arrives "since" the first watermark

        source = BiometricCandidateSource(modality="face")
        candidates = list(source.scan(Watermark(updated_at=first_watermark.updated_at, last_id=first_watermark.last_id)))

        # Only the newly-scanned template drives a scan iteration, but it still
        # matches against the full gallery (including the earlier template).
        self.assertEqual(len(candidates), 1)

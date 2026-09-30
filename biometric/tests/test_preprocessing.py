"""
Preprocessing tag (docs/wb-biometric-dedup-seam.md §6.13): the DeepFace
provider hands DeepFace a BGR array, every template records its provider's
preprocessing under metadata["preprocessing"], and verify(), identify(), the
impersonation probe and the dedup candidate source never compare two vectors
or templates whose preprocessing differs.
"""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import SimpleTestCase

from biometric.apps import BiometricConfig
from biometric.audit_chain import ACTION_VERIFY
from biometric.dedup_source import BiometricCandidateSource
from biometric.impersonation import PROBE_DEFAULTS
from biometric.models import BiometricAuditEvent, BiometricTemplate, BiometricVerification
from biometric.providers.base import Extracted
from biometric.providers.deepface_provider import DeepFaceProvider
from biometric.providers.fake import FakeEmbeddingProvider, FakeMatcherProvider, _hash_to_vector
from biometric.registry import ProviderRegistry
from biometric.services import (
    PREPROCESSING_KEY,
    PREPROCESSING_MISMATCH,
    enrol,
    identify,
    verify,
)
from biometric.tests.test_audit_chain import AuditConfigMixin
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase


def _png(pixels):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.array(pixels, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestDeepFaceColourOrder(SimpleTestCase):

    def _represent_input(self, sample):
        fake = MagicMock()
        fake.represent.return_value = [{"embedding": [0.1, 0.2], "facial_area": {}, "face_confidence": 0}]
        with patch("biometric.providers.deepface_provider._DeepFace", fake), \
                patch("biometric.providers.deepface_provider._DEEPFACE_AVAILABLE", True):
            DeepFaceProvider(detector_backend="retinaface").extract(sample)
        _, kwargs = fake.represent.call_args
        return kwargs["img_path"]

    def test_deepface_receives_bgr(self):
        red, green, blue = [255, 0, 0], [0, 255, 0], [0, 0, 255]
        array = self._represent_input(_png([[red, green], [blue, [10, 20, 30]]]))

        self.assertEqual(array[0][0].tolist(), [0, 0, 255])
        self.assertEqual(array[0][1].tolist(), [0, 255, 0])
        self.assertEqual(array[1][0].tolist(), [255, 0, 0])
        self.assertEqual(array[1][1].tolist(), [30, 20, 10])
        self.assertEqual(array.shape, (2, 2, 3))
        self.assertTrue(array.flags["C_CONTIGUOUS"])

    def test_grayscale_and_alpha_samples_become_three_bgr_channels(self):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGBA", (1, 1), (200, 100, 50, 7)).save(buffer, format="PNG")
        array = self._represent_input(buffer.getvalue())

        self.assertEqual(array[0][0].tolist(), [50, 100, 200])

    def test_the_provider_declares_its_preprocessing(self):
        self.assertEqual(DeepFaceProvider.preprocessing, "pillow_bgr")
        self.assertEqual(FakeEmbeddingProvider().preprocessing, "")


class TaggedEmbeddingProvider(FakeEmbeddingProvider):
    """The fake embedding provider, declaring a preprocessing tag."""

    preprocessing = "p1"

    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, "provider_name": "tagged_embedding"})


class TaggedMatcherProvider(FakeMatcherProvider):

    preprocessing = "p1"

    def __init__(self, **kwargs):
        super().__init__(**{**kwargs, "provider_name": "tagged_matcher"})


class _PreprocessingTestCase(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.modalities = {
            "face": {"provider": "tagged_embedding", "threshold": 0.68},
            "fingerprint": {"provider": "tagged_matcher", "threshold": 50.0},
        }
        BiometricConfig.impersonation_probe = dict(PROBE_DEFAULTS)
        BiometricConfig.vector_index = "numpy"
        BiometricConfig.dedup_threshold = {"face": 0.62, "fingerprint": 50.0}
        ProviderRegistry.register_modality("face", "tagged_embedding", TaggedEmbeddingProvider)
        ProviderRegistry.register_modality("fingerprint", "tagged_matcher", TaggedMatcherProvider)

    @staticmethod
    def _face(subject_id, sample, preprocessing, position=""):
        metadata = {PREPROCESSING_KEY: preprocessing} if preprocessing is not None else {}
        return BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id=subject_id, modality="face", kind="embedding",
            position=position, vector=_hash_to_vector(sample, 8), provider="tagged_embedding", model_name="",
            metadata=metadata,
        )

    @staticmethod
    def _print(subject_id, template, preprocessing):
        metadata = {PREPROCESSING_KEY: preprocessing} if preprocessing is not None else {}
        return BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id=subject_id, modality="fingerprint", kind="template",
            template=template, provider="tagged_matcher", model_name="", metadata=metadata,
        )


class TestEnrolRecordsThePreprocessing(_PreprocessingTestCase):

    def test_the_provider_tag_is_stored_in_metadata(self):
        row = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent", metadata={"cuvee_id": 3})

        row.refresh_from_db()
        self.assertEqual(row.metadata, {"cuvee_id": 3, PREPROCESSING_KEY: "p1"})

    def test_the_caller_cannot_set_the_tag(self):
        row = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent", metadata={PREPROCESSING_KEY: "forged"})

        row.refresh_from_db()
        self.assertEqual(row.metadata, {PREPROCESSING_KEY: "p1"})

    def test_a_device_template_gets_the_provider_tag(self):
        device = Extracted(vector=[0.1] * 8, metadata={PREPROCESSING_KEY: "forged", "device": "tab-1"})

        row = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent", device_template=device)

        row.refresh_from_db()
        self.assertEqual(row.metadata, {"device": "tab-1", PREPROCESSING_KEY: "p1"})

    def test_a_provider_without_a_tag_stores_none_and_drops_a_caller_tag(self):
        BiometricConfig.modalities = {"face": {"provider": "fake_embedding", "threshold": 0.68}}
        ProviderRegistry.register_modality("face", "fake_embedding", FakeEmbeddingProvider)

        row = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent", metadata={PREPROCESSING_KEY: "forged"})

        row.refresh_from_db()
        self.assertEqual(row.metadata, {})


class TestVerifySkipsOtherPreprocessing(_PreprocessingTestCase):

    def test_a_template_under_other_preprocessing_is_not_compared(self):
        self._face("s1", b"photo", "p0")

        with self.assertLogs("biometric.services", level="WARNING") as logs:
            result = verify(SUBJECT_MODEL, "s1", "face", sample=b"photo", actor="agent")

        self.assertIsNone(result.confidence)
        self.assertFalse(result.verified)
        self.assertEqual(result.template_skip_reason, PREPROCESSING_MISMATCH)
        row = BiometricVerification.objects.get()
        self.assertIsNone(row.score)
        self.assertEqual(row.template_skip_reason, PREPROCESSING_MISMATCH)
        self.assertIn("1 face template(s) skipped: preprocessing_mismatch", "\n".join(logs.output))

    def test_a_template_without_a_tag_is_not_compared_with_a_tagged_provider(self):
        self._face("s1", b"photo", None)

        result = verify(SUBJECT_MODEL, "s1", "face", sample=b"photo", actor="agent")

        self.assertIsNone(result.confidence)
        self.assertEqual(result.template_skip_reason, PREPROCESSING_MISMATCH)

    def test_the_matching_template_is_compared_and_the_skip_is_still_recorded(self):
        self._face("s1", b"other-photo", "p0", position="old")
        self._face("s1", b"photo", "p1", position="new")

        result = verify(SUBJECT_MODEL, "s1", "face", sample=b"photo", actor="agent")

        self.assertAlmostEqual(result.confidence, 1.0)
        self.assertTrue(result.verified)
        self.assertEqual(result.template_skip_reason, PREPROCESSING_MISMATCH)

    def test_nothing_skipped_records_no_reason(self):
        self._face("s1", b"photo", "p1")

        result = verify(SUBJECT_MODEL, "s1", "face", sample=b"photo", actor="agent")

        self.assertAlmostEqual(result.confidence, 1.0)
        self.assertEqual(result.template_skip_reason, "")
        self.assertEqual(BiometricVerification.objects.get().template_skip_reason, "")

    def test_template_modality(self):
        self._print("s1", b"minutiae", "p0")

        result = verify(SUBJECT_MODEL, "s1", "fingerprint", sample=b"minutiae", actor="agent")

        self.assertIsNone(result.confidence)
        self.assertEqual(result.template_skip_reason, PREPROCESSING_MISMATCH)

    def test_device_path_records_no_template_skip(self):
        BiometricConfig.modalities = {**BiometricConfig.modalities,
                                      "voice_device": {"provider": "device_reported", "threshold": 48}}
        BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id="s1", modality="voice_device", kind="template",
            template=b"voice", provider="device_reported", model_name="", metadata={PREPROCESSING_KEY: "p0"},
        )

        result = verify(SUBJECT_MODEL, "s1", "voice_device", device_score=60.0, actor="agent")

        self.assertTrue(result.verified)
        self.assertEqual(result.template_skip_reason, "")


class TestVerifyAuditAndGraphQL(AuditConfigMixin, _PreprocessingTestCase):

    def test_the_verify_event_carries_the_reason(self):
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        self._face("s1", b"photo", "p0")

        verify(SUBJECT_MODEL, "s1", "face", sample=b"photo", actor="agent")

        event = BiometricAuditEvent.objects.get(action=ACTION_VERIFY)
        self.assertEqual(event.payload.get("template_skip_reason"), PREPROCESSING_MISMATCH)
        self.assertIsNone(event.payload["score"])

    def test_graphql_exposes_the_reason_on_the_result_and_the_trail(self):
        import base64

        import graphene

        from biometric.schema import Mutation, Query

        self._face("s1", b"photo", "p0")
        user = MagicMock(is_anonymous=False, username="agent")
        user.has_perms.return_value = True
        schema = graphene.Schema(query=Query, mutation=Mutation)
        sample = base64.b64encode(b"photo").decode()

        result = schema.execute(
            'mutation { verifyBiometric(subjectId: "s1", modality: "face", sample: "%s") '
            "{ verified confidence templateSkipReason } }" % sample,
            context_value=SimpleNamespace(user=user, headers={}),
        )
        trail = schema.execute(
            'query { biometricVerifications(subjectId: "s1") { templateSkipReason } }',
            context_value=SimpleNamespace(user=user, headers={}),
        )

        self.assertIsNone(result.errors, result.errors)
        self.assertEqual(result.data["verifyBiometric"], {
            "verified": False, "confidence": None, "templateSkipReason": PREPROCESSING_MISMATCH,
        })
        self.assertIsNone(trail.errors, trail.errors)
        self.assertEqual(trail.data["biometricVerifications"], [{"templateSkipReason": PREPROCESSING_MISMATCH}])


class TestIdentifySkipsOtherPreprocessing(_PreprocessingTestCase):

    def test_embedding_gallery(self):
        self._face("old", b"photo", "p0")
        self._face("untagged", b"photo", None)
        kept = self._face("new", b"photo", "p1")

        with self.assertLogs("biometric.services", level="INFO") as logs:
            matches = identify("face", sample=b"photo", top_k=5)

        self.assertEqual([(m.subject_id, m.template_id) for m in matches], [("new", str(kept.id))])
        self.assertIn("2 face template(s) skipped: preprocessing_mismatch", "\n".join(logs.output))

    def test_a_skipped_row_never_takes_a_top_k_place(self):
        for index in range(3):
            self._face(f"old{index}", b"photo", "p0")
        self._face("new", b"another-photo", "p1")

        matches = identify("face", sample=b"photo", top_k=1)

        self.assertEqual([m.subject_id for m in matches], ["new"])

    def test_template_gallery(self):
        self._print("old", b"minutiae", "p0")
        self._print("new", b"minutiae", "p1")

        matches = identify("fingerprint", sample=b"minutiae", top_k=5)

        self.assertEqual([m.subject_id for m in matches], ["new"])

    def test_the_impersonation_probe_never_sees_a_skipped_row(self):
        BiometricConfig.impersonation_probe = {**PROBE_DEFAULTS, "enabled": True}
        self._face("s1", b"photo", "p1")
        self._face("twin-old", b"photo", "p0")

        result = verify(SUBJECT_MODEL, "s1", "face", sample=b"photo", actor="agent")

        self.assertEqual(result.impersonation.status, "ok")
        self.assertFalse(result.impersonation.suspected)
        self.assertEqual(result.impersonation.candidates, [])

        self._face("twin-new", b"photo", "p1")
        result = verify(SUBJECT_MODEL, "s1", "face", sample=b"photo", actor="agent")
        self.assertEqual([c["subject_id"] for c in result.impersonation.candidates], ["twin-new"])


class TestDedupSourceSkipsOtherPreprocessing(_PreprocessingTestCase):

    def test_a_probe_row_under_other_preprocessing_is_not_scanned(self):
        self._face("a", b"photo", "p0")
        self._face("b", b"photo", "p0")

        with self.assertLogs("biometric.dedup_source", level="INFO") as logs:
            candidates = list(BiometricCandidateSource("face").scan(None))

        self.assertEqual(candidates, [])
        self.assertIn("2 face template(s) skipped: preprocessing_mismatch", "\n".join(logs.output))

    def test_only_rows_under_the_provider_preprocessing_pair_up(self):
        self._face("a", b"photo", "p1")
        self._face("b", b"photo", "p0")
        self._face("c", b"photo", "p1")

        candidates = list(BiometricCandidateSource("face").scan(None))

        self.assertEqual([(c.subject_a, c.subject_b) for c in candidates], [("a", "c")])

    def test_nothing_skipped_logs_nothing(self):
        self._face("a", b"photo", "p1")

        with self.assertNoLogs("biometric.dedup_source", level="INFO"):
            list(BiometricCandidateSource("face").scan(None))

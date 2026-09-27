"""
Impersonation probe inside services.verify() (docs/wb-biometric-dedup-seam.md §6.9).

Vectors are preset per sample so that cosine similarities are known; the
hash-based fake gives all-positive vectors whose pairwise cosine is high.
"""

import json
import math
from unittest.mock import MagicMock, patch

from django.db import connection, transaction

from biometric import services
from biometric.apps import BiometricConfig
from biometric.impersonation import (
    PROBE_DEFAULTS, ImpersonationProbe, maybe_probe, probe_settings, probe_threshold,
)
from biometric.models import BiometricTemplate, BiometricVerification
from biometric.providers.base import Extracted
from biometric.providers.fake import FakeEmbeddingProvider
from biometric.registry import ProviderRegistry
from biometric.services import verify
from biometric.signals import IMPERSONATION_SUSPECTED
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase

VECTORS = {
    b"alice": [1.0, 0.0, 0.0],
    b"bob": [0.0, 1.0, 0.0],
    b"bob-probe": [0.05, 1.0, 0.0],
}


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


class PresetEmbeddingProvider(FakeEmbeddingProvider):
    """extract() returns the preset vector of the sample bytes."""

    def __init__(self, provider_name="preset_embedding", **kwargs):
        super().__init__(provider_name=provider_name, **kwargs)

    def extract(self, sample, position=None):
        return Extracted(vector=list(VECTORS[sample]))


class _ImpersonationTestCase(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.modalities = dict(BiometricConfig.modalities)
        BiometricConfig.modalities["face"] = {"provider": "preset_embedding", "threshold": 0.68}
        BiometricConfig.dedup_threshold = {"face": 0.62}
        BiometricConfig.vector_index = "numpy"
        BiometricConfig.impersonation_probe = dict(PROBE_DEFAULTS)
        ProviderRegistry.register_modality("face", "preset_embedding", PresetEmbeddingProvider)

    def _enable(self, **overrides):
        BiometricConfig.impersonation_probe = {**PROBE_DEFAULTS, "enabled": True, **overrides}

    @staticmethod
    def _face(subject_id, vector, *, subject_model=SUBJECT_MODEL, position=""):
        return BiometricTemplate.objects.create(
            subject_model=subject_model, subject_id=subject_id, modality="face", position=position,
            kind="embedding", vector=list(vector), provider="preset_embedding", model_name="",
        )

    @staticmethod
    def _template(subject_id, modality, provider, template):
        return BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id=subject_id, modality=modality,
            kind="template", template=template, provider=provider, model_name="",
        )

    @staticmethod
    def _row():
        return BiometricVerification.objects.order_by("-created_at").first()

    def _bind_receiver(self, receiver):
        from core.service_signals import ServiceSignalBindType
        from core.signals import REGISTERED_SERVICE_SIGNALS, bind_service_signal

        bind_service_signal(IMPERSONATION_SUSPECTED, receiver, bind_type=ServiceSignalBindType.AFTER)
        registered = REGISTERED_SERVICE_SIGNALS[IMPERSONATION_SUSPECTED]

        def disconnect():
            registered.after_service_signal.disconnect(receiver)
            if receiver in registered.connected_signals["after"]:
                registered.connected_signals["after"].remove(receiver)

        self.addCleanup(disconnect)


class TestDefaultUnchanged(_ImpersonationTestCase):

    def test_shipped_default_does_not_probe_and_keeps_query_count(self):
        self.assertFalse(BiometricConfig.impersonation_probe["enabled"])
        self._face("alice", [1.0, 0.0, 0.0])

        with patch("biometric.services.identify") as identify:
            with self.assertNumQueries(2):
                result = verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester")

        identify.assert_not_called()
        self.assertIsNone(result.impersonation)
        row = self._row()
        self.assertEqual(row.impersonation_status, "")
        self.assertFalse(row.impersonation_suspected)
        self.assertEqual(row.impersonation_subject_model, "")
        self.assertEqual(row.impersonation_subject_id, "")
        self.assertIsNone(row.impersonation_score)
        self.assertEqual(row.impersonation_evidence, {})

    def test_device_path_never_probes(self):
        self._enable(modalities=["face", "voice_device"])

        with patch("biometric.services.identify") as identify:
            with self.assertNumQueries(1):
                result = verify(SUBJECT_MODEL, "alice", "voice_device", device_score=60, actor="tester")

        identify.assert_not_called()
        self.assertIsNone(result.impersonation)


class TestSettings(_ImpersonationTestCase):

    def test_enabled_alone_takes_the_other_defaults(self):
        BiometricConfig.impersonation_probe = {"enabled": True}

        settings = probe_settings()
        self.assertEqual(settings["modalities"], ["face"])
        self.assertEqual(settings["top_k"], 5)
        self.assertIsNone(settings["margin"])
        self.assertEqual(settings["thresholds"], {})

        self._face("alice", [1.0, 0.0, 0.0])
        with patch("biometric.services.identify", wraps=services.identify) as identify:
            result = verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester")
        identify.assert_called_once()
        self.assertEqual(result.impersonation.status, "ok")

    def test_empty_or_none_setting_keeps_the_probe_off(self):
        self._face("alice", [1.0, 0.0, 0.0])
        for configured in ({}, None):
            with self.subTest(configured=configured):
                BiometricConfig.impersonation_probe = configured
                with patch("biometric.services.identify") as identify:
                    result = verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester")
                identify.assert_not_called()
                self.assertIsNone(result.impersonation)

    def test_modality_not_listed_does_not_probe(self):
        self._enable(modalities=["fingerprint"])
        self._face("alice", [1.0, 0.0, 0.0])

        with patch("biometric.services.identify") as identify:
            result = verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester")

        identify.assert_not_called()
        self.assertIsNone(result.impersonation)


class TestSuspicion(_ImpersonationTestCase):

    def setUp(self):
        super().setUp()
        self._face("alice", VECTORS[b"alice"])
        self.bob = self._face("bob", VECTORS[b"bob"])

    def test_foreign_match_raises_a_suspicion(self):
        self._enable()

        result = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")

        probe = result.impersonation
        bob_score = _cosine(VECTORS[b"bob-probe"], VECTORS[b"bob"])
        self.assertEqual(probe.status, "ok")
        self.assertTrue(probe.suspected)
        self.assertEqual(probe.best_match["subject_id"], "bob")
        self.assertEqual(probe.best_match["template_id"], str(self.bob.id))
        self.assertAlmostEqual(probe.best_match["score"], bob_score, places=6)
        self.assertEqual(probe.threshold, 0.62)

        row = self._row()
        self.assertEqual(row.impersonation_status, "ok")
        self.assertTrue(row.impersonation_suspected)
        self.assertEqual(row.impersonation_subject_model, SUBJECT_MODEL)
        self.assertEqual(row.impersonation_subject_id, "bob")
        self.assertAlmostEqual(row.impersonation_score, bob_score, places=6)
        self.assertEqual(row.impersonation_evidence["threshold"], 0.62)
        self.assertIsNotNone(row.impersonation_evidence["latency_ms"])

    def test_probe_leaves_verdict_score_and_threshold_untouched(self):
        off = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")
        off_row = self._row()
        self._enable()
        on = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")
        on_row = self._row()

        self.assertIsNone(off.impersonation)
        self.assertTrue(on.impersonation.suspected)
        self.assertEqual(on.verified, off.verified)
        self.assertEqual(on.confidence, off.confidence)
        self.assertEqual(on.threshold, off.threshold)
        self.assertEqual(on_row.score, off_row.score)
        self.assertEqual(on_row.verified, off_row.verified)
        self.assertEqual(on_row.threshold, off_row.threshold)

    def test_claimed_subject_is_excluded_and_its_score_captured(self):
        self._enable()

        result = verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester")

        probe = result.impersonation
        self.assertFalse(probe.suspected)
        self.assertEqual(probe.candidates, [])
        self.assertAlmostEqual(probe.claimed_score, 1.0, places=6)
        self.assertAlmostEqual(self._row().impersonation_evidence["claimed_score"], 1.0, places=6)

    def test_claimed_subject_never_appears_among_candidates(self):
        self._enable(thresholds={"face": -1.0}, margin=0.05)

        probe = verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester").impersonation

        self.assertEqual([(c["subject_model"], c["subject_id"]) for c in probe.candidates], [(SUBJECT_MODEL, "bob")])
        self.assertFalse(probe.suspected)

    def test_exclusion_is_by_subject_model_and_subject_id_pair(self):
        self._enable()
        self._face("alice", VECTORS[b"bob"], subject_model="insuree.Insuree")
        BiometricTemplate.objects.filter(subject_id="bob").delete()

        probe = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester").impersonation

        self.assertTrue(probe.suspected)
        self.assertEqual(
            (probe.best_match["subject_model"], probe.best_match["subject_id"]), ("insuree.Insuree", "alice"),
        )

    def test_threshold_filters_foreign_matches(self):
        self._enable(thresholds={"face": 0.999})

        probe = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester").impersonation

        self.assertEqual(probe.threshold, 0.999)
        self.assertEqual(probe.candidates, [])
        self.assertFalse(probe.suspected)
        self.assertFalse(self._row().impersonation_suspected)

    def test_per_subject_collapse_keeps_the_best_template(self):
        self._enable()
        second = [0.1, 1.0, 0.0]
        self._face("bob", second, position="left")

        probe = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester").impersonation

        bob_rows = [c for c in probe.candidates if c["subject_id"] == "bob"]
        self.assertEqual(len(bob_rows), 1)
        expected = max(_cosine(VECTORS[b"bob-probe"], VECTORS[b"bob"]), _cosine(VECTORS[b"bob-probe"], second))
        self.assertAlmostEqual(bob_rows[0]["score"], expected, places=6)

    def test_risk_profile_does_not_change_the_probe_threshold_or_verdict(self):
        self._enable()
        BiometricConfig.risk_profiles = {"strict": {"modality_thresholds": {"face": 0.9}}}

        base = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")
        strict = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester", risk_profile="strict")

        self.assertEqual(strict.threshold, 0.9)
        self.assertEqual(base.impersonation.threshold, 0.62)
        self.assertEqual(strict.impersonation.threshold, 0.62)
        self.assertEqual(strict.impersonation.suspected, base.impersonation.suspected)
        self.assertEqual(strict.impersonation.best_match, base.impersonation.best_match)


class TestTopK(_ImpersonationTestCase):

    def test_reads_top_k_plus_claimed_templates(self):
        self._enable(top_k=1)
        for position, vector in (("p1", [1.0, 0.0, 0.0]), ("p2", [0.99, 0.1, 0.0]), ("p3", [0.98, 0.2, 0.0])):
            self._face("alice", vector, position=position)
        self._face("carol", [0.9, 0.3, 0.0])
        self._face("dave", [0.7, 0.7, 0.0])

        with patch("biometric.services.identify", wraps=services.identify) as identify:
            probe = verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester").impersonation

        self.assertEqual(identify.call_args.kwargs["top_k"], 4)
        self.assertNotIn("sample", identify.call_args.kwargs)
        self.assertNotIn("actor", identify.call_args.kwargs)
        self.assertEqual([c["subject_id"] for c in probe.candidates], ["carol"])
        self.assertTrue(probe.suspected)


class TestThresholdChain(_ImpersonationTestCase):

    def test_each_link_of_the_fallback_chain(self):
        provider = FakeEmbeddingProvider(default_threshold=0.3)

        self.assertEqual(probe_threshold("face", provider, {"thresholds": {"face": 0.5}}), 0.5)
        self.assertEqual(probe_threshold("face", provider, {"thresholds": {}}), 0.62)

        BiometricConfig.dedup_threshold = {}
        self.assertEqual(probe_threshold("face", provider, {"thresholds": {}}), 0.68)

        BiometricConfig.modalities = {**BiometricConfig.modalities, "face": {"provider": "preset_embedding"}}
        self.assertEqual(probe_threshold("face", provider, {"thresholds": {}}), 0.3)


class TestMargin(_ImpersonationTestCase):

    def setUp(self):
        super().setUp()
        self._face("alice", [1.0, 0.0, 0.0])

    def test_margin_none_flags_a_foreign_match_far_below_the_claimed_score(self):
        self._face("bob", [0.8, 0.6, 0.0])
        self._enable(margin=None)

        probe = verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester").impersonation

        self.assertAlmostEqual(probe.claimed_score, 1.0, places=6)
        self.assertAlmostEqual(probe.best_match["score"], 0.8, places=6)
        self.assertTrue(probe.suspected)

    def test_margin_excludes_a_foreign_match_outside_it(self):
        self._face("bob", [0.8, 0.6, 0.0])
        self._enable(margin=0.05)

        probe = verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester").impersonation

        self.assertFalse(probe.suspected)
        self.assertIsNone(probe.best_match)
        self.assertEqual([(c["subject_id"], c["suspect"]) for c in probe.candidates], [("bob", False)])

    def test_margin_flags_a_foreign_match_within_it(self):
        self._face("bob", [0.99, 0.141, 0.0])
        self._enable(margin=0.05)

        probe = verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester").impersonation

        self.assertTrue(probe.suspected)
        self.assertEqual(probe.best_match["subject_id"], "bob")

    def test_margin_flags_when_the_claimed_subject_has_no_template(self):
        self._face("bob", [0.8, 0.6, 0.0])
        self._enable(margin=0.05)

        probe = verify(SUBJECT_MODEL, "zed", "face", sample=b"alice", actor="tester").impersonation

        self.assertIsNone(probe.claimed_score)
        self.assertTrue(probe.suspected)
        self.assertEqual(probe.best_match["subject_id"], "alice")


class TestFailureIsolation(_ImpersonationTestCase):

    def setUp(self):
        super().setUp()
        self._face("alice", VECTORS[b"alice"])
        self._face("bob", VECTORS[b"bob"])
        self.fired = []
        self._bind_receiver(self._receiver)

    def _receiver(self, **kwargs):
        self.fired.append(kwargs)

    def test_probe_failure_is_recorded_and_never_raises(self):
        baseline = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")
        self._enable()

        with patch("biometric.services.identify", side_effect=RuntimeError("index unavailable")):
            with self.captureOnCommitCallbacks(execute=True):
                result = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")

        self.assertEqual(result.verified, baseline.verified)
        self.assertEqual(result.confidence, baseline.confidence)
        self.assertEqual(result.impersonation.status, "failed")
        self.assertFalse(result.impersonation.suspected)
        row = self._row()
        self.assertEqual(row.impersonation_status, "failed")
        self.assertFalse(row.impersonation_suspected)
        self.assertEqual(row.impersonation_evidence["error"], "RuntimeError: impersonation probe failed")
        self.assertNotIn("index unavailable", row.impersonation_evidence["error"])
        self.assertEqual(self.fired, [])

    def test_database_error_in_probe_does_not_poison_the_verification_row(self):
        self._enable()

        def broken_identify(*args, **kwargs):
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM biometric_no_such_table")

        with transaction.atomic():
            with patch("biometric.services.identify", side_effect=broken_identify):
                result = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")
            row = BiometricVerification.objects.get(subject_id="alice")

        self.assertEqual(result.impersonation.status, "failed")
        self.assertEqual(row.impersonation_status, "failed")
        self.assertEqual(row.impersonation_evidence["error"], "ProgrammingError: impersonation probe failed")
        self.assertNotIn("biometric_no_such_table", row.impersonation_evidence["error"])


class TestFailureRedaction(_ImpersonationTestCase):
    """A failed probe records and logs the exception class only, never its message."""

    SECRET = "gAAAAABstored-ciphertext-of-another-subject"

    @staticmethod
    def _gql(probe_or_row):
        from biometric.schema import _impersonation_gql

        user = MagicMock()
        user.has_perms.return_value = True
        return _impersonation_gql(probe_or_row, user)

    def _assert_logs_exclude(self, logs, secret):
        for record in logs.records:
            self.assertNotIn(secret, record.getMessage())
            self.assertIsNone(record.exc_info)

    def test_exception_message_is_kept_out_of_evidence_graphql_and_logs(self):
        self._enable()
        provider = ProviderRegistry.get_provider("face")
        leak = ValueError(f"could not convert string to float: '{self.SECRET}'")

        with self.assertLogs("biometric.impersonation", level="DEBUG") as logs:
            with patch("biometric.services.identify", side_effect=leak):
                probe = maybe_probe(SUBJECT_MODEL, "alice", "face", provider, Extracted(vector=[1.0, 0.0, 0.0]))

        self.assertEqual(probe.status, "failed")
        self.assertEqual(probe.error, "ValueError: impersonation probe failed")
        self.assertNotIn(self.SECRET, json.dumps(probe.as_evidence()))
        self.assertEqual(self._gql(probe).error, "ValueError: impersonation probe failed")
        self._assert_logs_exclude(logs, self.SECRET)

    def test_undecryptable_gallery_ciphertext_never_reaches_the_row(self):
        from cryptography.fernet import Fernet

        from biometric import crypto

        ciphertext = crypto.encrypt_vector(VECTORS[b"bob"], Fernet.generate_key().decode())
        BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id="bob", modality="face", kind="embedding",
            vector=ciphertext, encrypted=True, provider="preset_embedding", model_name="",
        )
        BiometricConfig.template_key = Fernet.generate_key().decode()
        with self.assertRaises(ValueError) as raw:
            services.identify("face", vector=VECTORS[b"bob-probe"], top_k=5)
        self.assertIn(ciphertext, str(raw.exception))
        self._enable()

        with self.assertLogs("biometric.impersonation", level="DEBUG") as logs:
            verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")

        row = self._row()
        self.assertEqual(row.impersonation_status, "failed")
        self.assertEqual(row.impersonation_evidence["error"], "ValueError: impersonation probe failed")
        self.assertNotIn(ciphertext, json.dumps(row.impersonation_evidence))
        self.assertEqual(self._gql(row).error, "ValueError: impersonation probe failed")
        self._assert_logs_exclude(logs, ciphertext)


class TestTemplateModalities(_ImpersonationTestCase):

    def test_exact_template_on_another_subject_is_suspected(self):
        self._enable(modalities=["fingerprint"])
        self._template("alice", "fingerprint", "fake_matcher", b"print-alice")
        self._template("bob", "fingerprint", "fake_matcher", b"print-bob")

        result = verify(SUBJECT_MODEL, "alice", "fingerprint", sample=b"print-bob", actor="tester")

        self.assertFalse(result.verified)
        self.assertTrue(result.impersonation.suspected)
        self.assertEqual(result.impersonation.threshold, 50.0)
        self.assertEqual(result.impersonation.best_match["subject_id"], "bob")
        self.assertEqual(result.impersonation.best_match["score"], 100.0)
        self.assertEqual(self._row().impersonation_score, 100.0)

    def test_device_reported_modality_records_a_failed_probe(self):
        self._enable(modalities=["voice_device"])
        self._template("bob", "voice_device", "device_reported", b"device-template")

        result = verify(SUBJECT_MODEL, "alice", "voice_device", sample=b"device-template", actor="tester")

        self.assertIsNone(result.confidence)
        self.assertEqual(result.impersonation.status, "failed")
        row = self._row()
        self.assertEqual(row.impersonation_status, "failed")
        self.assertEqual(row.impersonation_evidence["error"], "NotImplementedError: impersonation probe failed")
        self.assertNotIn("device_reported", row.impersonation_evidence["error"])


class TestImpersonationSignal(_ImpersonationTestCase):

    def setUp(self):
        super().setUp()
        self._face("alice", VECTORS[b"alice"])
        self.bob = self._face("bob", VECTORS[b"bob"])
        self._enable()
        self.fired = []

    def _receiver(self, **kwargs):
        self.fired.append(kwargs)

    def test_registered_at_startup(self):
        from core.signals import REGISTERED_SERVICE_SIGNALS

        self.assertTrue(REGISTERED_SERVICE_SIGNALS[IMPERSONATION_SUSPECTED].is_signal_registered())

    def test_fires_once_after_commit_with_the_payload(self):
        self._bind_receiver(self._receiver)

        with self.captureOnCommitCallbacks(execute=False) as callbacks:
            verify(
                SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester",
                device_id="tab-7", context={"site": "koza"},
            )
        self.assertEqual(self.fired, [])
        for callback in callbacks:
            callback()

        self.assertEqual(len(self.fired), 1)
        payload = self.fired[0]["result"]
        row = self._row()
        self.assertEqual(payload["verification_id"], str(row.id))
        self.assertEqual(payload["subject_model"], SUBJECT_MODEL)
        self.assertEqual(payload["subject_id"], "alice")
        self.assertEqual(payload["modality"], "face")
        self.assertEqual(payload["matched_subject_model"], SUBJECT_MODEL)
        self.assertEqual(payload["matched_subject_id"], "bob")
        self.assertEqual(payload["matched_template_id"], str(self.bob.id))
        self.assertAlmostEqual(payload["matched_score"], _cosine(VECTORS[b"bob-probe"], VECTORS[b"bob"]), places=6)
        self.assertAlmostEqual(payload["claimed_score"], _cosine(VECTORS[b"bob-probe"], VECTORS[b"alice"]), places=6)
        self.assertEqual(payload["threshold"], 0.62)
        self.assertIsNone(payload["margin"])
        self.assertEqual(payload["actor"], "tester")
        self.assertEqual(payload["device_id"], "tab-7")
        self.assertEqual(payload["context"], {"site": "koza"})

    def test_fires_under_execute_true_only_on_suspicion(self):
        self._bind_receiver(self._receiver)

        with self.captureOnCommitCallbacks(execute=True):
            verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="tester")
        self.assertEqual(self.fired, [])

        with self.captureOnCommitCallbacks(execute=True):
            verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")
        self.assertEqual(len(self.fired), 1)

    def test_raising_receiver_does_not_break_verify(self):
        def raising_receiver(**kwargs):
            raise RuntimeError("subscriber down")

        self._bind_receiver(raising_receiver)

        with self.captureOnCommitCallbacks(execute=True):
            result = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")

        self.assertTrue(result.impersonation.suspected)
        self.assertTrue(BiometricVerification.objects.filter(subject_id="alice", impersonation_suspected=True).exists())

    def test_result_probe_type(self):
        result = verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="tester")

        self.assertIsInstance(result.impersonation, ImpersonationProbe)
        evidence = result.impersonation.as_evidence()
        self.assertEqual(
            set(evidence), {"threshold", "margin", "top_k", "claimed_score", "candidates", "error", "latency_ms"},
        )

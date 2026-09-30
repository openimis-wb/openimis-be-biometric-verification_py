"""
Audit events recorded by the service producers (docs/wb-biometric-dedup-seam.md §6.10):
enrol, verify (with the risk profile and the impersonation probe), identify,
consolidate, templates_of and purge; and the default path with audit off.
"""

import base64
import json
from datetime import timedelta
from unittest.mock import patch

from cryptography.fernet import Fernet
from django.db import connection
from django.test.utils import CaptureQueriesContext
from django.utils import timezone

from biometric import services
from biometric.apps import BiometricConfig
from biometric.audit_chain import (
    ACTION_CONSOLIDATE,
    ACTION_ENROL,
    ACTION_ENROL_REFUSED,
    ACTION_IDENTIFY,
    ACTION_IMPERSONATION,
    ACTION_PURGE,
    ACTION_TEMPLATE_READ,
    ACTION_VERIFY,
    FORBIDDEN_PAYLOAD_KEYS,
    compute_hash,
    verify_chain,
)
from biometric.dedup_source import BiometricCandidateSource
from biometric.models import (
    BiometricAlert,
    BiometricAuditEvent,
    BiometricErasure,
    BiometricRetentionPolicy,
    BiometricTemplate,
    BiometricVerification,
)
from biometric.providers.base import Extracted, FaceGeometry
from biometric.services import (
    QualityRefusedError,
    UnknownRiskProfileError,
    consolidate,
    enrol,
    identify,
    purge,
    record_impersonation_suspected,
    templates_of,
    verify,
)
from biometric.tests.test_audit_chain import restore_audit_settings_on_cleanup
from biometric.tests.test_impersonation import VECTORS, _ImpersonationTestCase
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase
from biometric.tests.synthetic_subjects import SyntheticSubjectsMixin

AUDIT_ON = {"enabled": True, "rules": {}}


def _keys(value):
    if isinstance(value, dict):
        found = set(value)
        for item in value.values():
            found |= _keys(item)
        return found
    if isinstance(value, list):
        found = set()
        for item in value:
            found |= _keys(item)
        return found
    return set()


def _strings(value):
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)] + [k for k in value]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return [value] if isinstance(value, str) else []


def _events(action=None):
    qs = BiometricAuditEvent.objects.order_by("sequence")
    return list(qs.filter(action=action) if action else qs)


class _AuditServiceTestCase(_MultimodalServiceTestCase):

    def setUp(self):
        restore_audit_settings_on_cleanup(self)
        super().setUp()
        BiometricConfig.audit = dict(AUDIT_ON)


class TestEnrolEvent(SyntheticSubjectsMixin, _AuditServiceTestCase):

    def test_records_template_enrol_with_superseded_ids(self):
        first = enrol(SUBJECT_MODEL, "s1", "face", b"photo-1", actor="agent")
        second = enrol(SUBJECT_MODEL, "s1", "face", b"photo-2", actor="agent")

        events = _events(ACTION_ENROL)
        self.assertEqual(len(events), 2)
        event = events[1]
        self.assertEqual((event.actor, event.subject_model, event.subject_id, event.modality),
                         ("agent", SUBJECT_MODEL, "s1", "face"))
        self.assertEqual(events[0].payload["template_id"], str(first.id))
        self.assertEqual(events[0].payload["superseded"], [])
        self.assertEqual(event.payload["template_id"], str(second.id))
        self.assertEqual(event.payload["superseded"], [str(first.id)])
        self.assertEqual(event.payload["provider"], "fake_embedding")
        self.assertEqual(event.payload["kind"], "embedding")
        self.assertFalse(event.payload["encrypted"])
        self.assertFalse(event.payload["device_template"])
        self.assertEqual(event.payload["quality_status"], second.quality_status)
        self.assertIsInstance(event.payload["quality_reasons"], list)

    def test_payload_holds_no_sample_and_no_vector(self):
        template = enrol(SUBJECT_MODEL, "s1", "face", b"photo-bytes", actor="agent")
        dumped = json.dumps(_events(ACTION_ENROL)[0].payload)

        self.assertNotIn(repr(b"photo-bytes"), dumped)
        self.assertNotIn("photo-bytes", dumped)
        self.assertNotIn(base64.b64encode(b"photo-bytes").decode(), dumped)
        self.assertNotIn(json.dumps(template.vector), dumped)
        for value in template.vector:
            self.assertNotIn(repr(value), dumped)
        self.assertFalse(_keys(_events(ACTION_ENROL)[0].payload) & FORBIDDEN_PAYLOAD_KEYS)

    def test_device_template_flag(self):
        enrol(SUBJECT_MODEL, "s1", "voice_device", sample=b"unused", actor="agent",
              device_template=Extracted(template=b"device-supplied"))
        self.assertTrue(_events(ACTION_ENROL)[0].payload["device_template"])
        self.assertNotIn("device-supplied", json.dumps(_events(ACTION_ENROL)[0].payload))

    def test_enforce_refusal_records_one_refusal_event_and_no_template(self):
        BiometricConfig.quality = {"mode": "enforce", "modalities": {}}
        device = Extracted(vector=[0.1] * 8, face=FaceGeometry(pose={"roll": 30.0}))

        with self.assertRaises(QualityRefusedError):
            enrol(SUBJECT_MODEL, "s1", "face", b"raw", actor="agent", device_template=device)

        self.assertEqual(BiometricAuditEvent.objects.count(), 1)
        self.assertEqual(_events(ACTION_ENROL), [])
        self.assertFalse(BiometricTemplate.objects.filter(subject_id="s1").exists())
        event = _events(ACTION_ENROL_REFUSED)[0]
        self.assertEqual((event.actor, event.subject_model, event.subject_id, event.modality),
                         ("agent", SUBJECT_MODEL, "s1", "face"))
        self.assertEqual(event.payload["quality_status"], "REFUSED")
        self.assertEqual(event.payload["quality_mode"], "enforce")
        self.assertEqual(event.payload["quality_reasons"], ["roll_above_max"])
        roll = [m for m in event.payload["quality_measures"] if m["name"] == "roll"][0]
        self.assertEqual((roll["value"], roll["limit"], roll["passed"], roll["source"]),
                         (30.0, 20.0, False, "provider_pose"))
        self.assertTrue(event.payload["device_template"])
        self.assertEqual(event.payload["provider"], "fake_embedding")
        self.assertEqual(event.payload["kind"], "embedding")
        self.assertEqual(event.payload["position"], "")
        self.assertFalse(_keys(event.payload) & FORBIDDEN_PAYLOAD_KEYS)
        self.assertNotIn("0.1", [str(v) for v in _strings(event.payload)])

    def test_server_extracted_image_refusal_records_the_measures(self):
        import io

        import numpy as np
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(buffer, format="PNG")
        BiometricConfig.quality = {"mode": "enforce"}

        with self.assertRaises(QualityRefusedError) as raised:
            enrol(SUBJECT_MODEL, "s1", "face", buffer.getvalue(), actor="agent")

        event = _events(ACTION_ENROL_REFUSED)[0]
        self.assertEqual(event.payload["quality_reasons"], ["sharpness_below_min"])
        self.assertFalse(event.payload["device_template"])
        self.assertEqual(event.payload["quality_measures"], raised.exception.verdict.as_dict()["measures"])
        sharp = [m for m in event.payload["quality_measures"] if m["name"] == "sharpness"][0]
        self.assertEqual((sharp["value"], sharp["passed"]), (0.0, False))
        self.assertEqual(event.hash, compute_hash(event, event.prev_hash))
        self.assertTrue(verify_chain().ok)

    def test_refusal_keeps_the_previous_active_template(self):
        kept = enrol(SUBJECT_MODEL, "s1", "face", b"photo-1", actor="agent")
        BiometricConfig.quality = {"mode": "enforce", "modalities": {}}
        device = Extracted(vector=[0.1] * 8, face=FaceGeometry(pose={"roll": 30.0}))

        with self.assertRaises(QualityRefusedError):
            enrol(SUBJECT_MODEL, "s1", "face", b"raw", actor="agent", device_template=device)

        kept.refresh_from_db()
        self.assertIsNone(kept.validity_to)
        self.assertEqual([e.action for e in _events()], [ACTION_ENROL, ACTION_ENROL_REFUSED])

    def test_refusal_over_graphql_records_the_event(self):
        import io
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import graphene
        import numpy as np
        from PIL import Image

        from biometric.schema import Mutation, Query

        buffer = io.BytesIO()
        Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(buffer, format="PNG")
        BiometricConfig.quality = {"mode": "enforce"}
        user = MagicMock(is_anonymous=False, username="agent")
        user.has_perms.return_value = True
        query = 'mutation { enrolBiometric(subjectId: "s1", modality: "face", sample: "%s") { id } }' % (
            base64.b64encode(buffer.getvalue()).decode()
        )

        schema = graphene.Schema(query=Query, mutation=Mutation)
        result = schema.execute(query, context_value=SimpleNamespace(user=user, headers={}))

        self.assertIsNone(result.data["enrolBiometric"])
        self.assertEqual(result.errors[0].extensions["code"], "BIOMETRIC_QUALITY_REFUSED")
        self.assertEqual(len(_events(ACTION_ENROL_REFUSED)), 1)
        self.assertEqual(_events(ACTION_ENROL_REFUSED)[0].actor, "agent")

    def test_event_failure_rolls_back_the_enrolment(self):
        with patch("biometric.audit_chain.record_event", side_effect=RuntimeError("chain down")):
            with self.assertRaises(RuntimeError):
                enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent")
        self.assertFalse(BiometricTemplate.objects.filter(subject_id="s1").exists())


class TestVerifyEvent(_AuditServiceTestCase):

    def test_server_path(self):
        enrol(SUBJECT_MODEL, "s1", "face", b"reference", actor="agent")

        result = verify(SUBJECT_MODEL, "s1", "face", sample=b"reference", actor="agent",
                        device_id="tab-1", context={"site": "koza", "note": "free text"})

        event = _events(ACTION_VERIFY)[0]
        row = BiometricVerification.objects.get(subject_id="s1")
        self.assertEqual(event.payload["verification_id"], str(row.id))
        self.assertTrue(event.payload["verified"])
        self.assertAlmostEqual(event.payload["score"], result.confidence, places=9)
        self.assertEqual(event.payload["threshold"], result.threshold)
        self.assertEqual(event.payload["origin"], "server")
        self.assertEqual(event.payload["device_id"], "tab-1")
        self.assertEqual(event.payload["risk_profile"], "")
        self.assertEqual(event.payload["impersonation_status"], "")
        self.assertFalse(event.payload["impersonation_suspected"])
        self.assertNotIn("context", event.payload)
        self.assertNotIn("site", _keys(event.payload))
        self.assertNotIn("free text", _strings(event.payload))

    def test_device_path_and_risk_profile(self):
        BiometricConfig.risk_profiles = {"voice_strict": {"modality_thresholds": {"voice_device": 70}}}

        verify(SUBJECT_MODEL, "s1", "voice_device", device_score=60.0, actor="agent", risk_profile="voice_strict")

        event = _events(ACTION_VERIFY)[0]
        self.assertEqual(event.payload["origin"], "device")
        self.assertEqual(event.payload["risk_profile"], "voice_strict")
        self.assertEqual(event.payload["threshold"], 70)
        self.assertFalse(event.payload["verified"])

    def test_negative_zero_device_score_keeps_the_chain_intact(self):
        verify(SUBJECT_MODEL, "s1", "voice_device", device_score=-0.0, actor="agent")
        verify(SUBJECT_MODEL, "s2", "voice_device", device_score=60.0, actor="agent")

        event = _events(ACTION_VERIFY)[0]
        self.assertEqual(event.payload["score"], 0.0)
        self.assertEqual(event.hash, compute_hash(event, event.prev_hash))
        report = verify_chain()
        self.assertTrue(report.ok, report.divergence)
        self.assertEqual(report.checked, 2)

    def test_unknown_risk_profile_records_nothing(self):
        with self.assertRaises(UnknownRiskProfileError):
            verify(SUBJECT_MODEL, "s1", "voice_device", device_score=60.0, actor="agent", risk_profile="nope")
        self.assertEqual(BiometricAuditEvent.objects.count(), 0)

    def test_event_failure_rolls_back_the_verification_row(self):
        with patch("biometric.audit_chain.record_event", side_effect=RuntimeError("chain down")):
            with self.assertRaises(RuntimeError):
                verify(SUBJECT_MODEL, "s1", "voice_device", device_score=60.0, actor="agent")
        self.assertFalse(BiometricVerification.objects.filter(subject_id="s1").exists())

    def test_fernet_ciphertext_never_reaches_a_payload(self):
        BiometricConfig.template_key = Fernet.generate_key()
        enrol(SUBJECT_MODEL, "s1", "face", b"reference", actor="agent")
        enrol(SUBJECT_MODEL, "s1", "fingerprint", b"minutiae", actor="agent")
        verify(SUBJECT_MODEL, "s1", "face", sample=b"reference", actor="agent")
        identify("face", sample=b"reference", actor="agent")
        templates_of(SUBJECT_MODEL, "s1", actor="agent")
        consolidate(SUBJECT_MODEL, "kept", "s1", actor="agent")

        self.assertGreaterEqual(BiometricAuditEvent.objects.count(), 6)
        for event in _events():
            self.assertFalse([s for s in _strings(event.payload) if s.startswith("gAAAA")], event.action)


class TestImpersonationEvents(_ImpersonationTestCase):

    def setUp(self):
        restore_audit_settings_on_cleanup(self)
        super().setUp()
        BiometricConfig.audit = dict(AUDIT_ON)
        self._face("alice", VECTORS[b"alice"])
        self.bob = self._face("bob", VECTORS[b"bob"])
        self._enable()

    def test_verify_then_impersonation_event_with_aligned_names(self):
        verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="agent")

        events = _events()
        self.assertEqual([e.action for e in events], [ACTION_VERIFY, ACTION_IMPERSONATION])
        row = self._row()
        verify_event, suspicion = events
        self.assertTrue(verify_event.payload["impersonation_suspected"])
        self.assertEqual(verify_event.payload["impersonation_status"], "ok")
        self.assertEqual((suspicion.subject_model, suspicion.subject_id, suspicion.modality, suspicion.actor),
                         (SUBJECT_MODEL, "alice", "face", "agent"))
        self.assertEqual(set(suspicion.payload), {
            "verification_id", "matched_subject_model", "matched_subject_id", "matched_template_id",
            "matched_score", "claimed_score", "threshold", "margin",
        })
        self.assertEqual(suspicion.payload["verification_id"], str(row.id))
        self.assertEqual(suspicion.payload["matched_subject_model"], SUBJECT_MODEL)
        self.assertEqual(suspicion.payload["matched_subject_id"], "bob")
        self.assertEqual(suspicion.payload["matched_template_id"], str(self.bob.id))
        self.assertAlmostEqual(suspicion.payload["matched_score"], row.impersonation_score, places=9)
        self.assertEqual(suspicion.payload["threshold"], 0.62)
        self.assertIsNone(suspicion.payload["margin"])

    def test_one_impersonation_event_even_after_the_signal_fires(self):
        with self.captureOnCommitCallbacks(execute=True):
            verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="agent")

        self.assertEqual(len(_events(ACTION_IMPERSONATION)), 1)
        self.assertEqual(BiometricAlert.objects.filter(rule_kind="IMPERSONATION_SUSPECTED").count(), 1)

    def test_no_suspicion_records_verify_only(self):
        verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", actor="agent")
        self.assertEqual([e.action for e in _events()], [ACTION_VERIFY])

    def test_probe_queries_run_before_the_lock(self):
        with CaptureQueriesContext(connection) as queries:
            verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="agent")

        sql = [q["sql"] for q in queries.captured_queries]
        lock_at = next(i for i, s in enumerate(sql) if "pg_advisory_xact_lock" in s)
        gallery_reads = [i for i, s in enumerate(sql) if 'FROM "biometric_template"' in s]
        self.assertTrue(gallery_reads)
        self.assertLess(max(gallery_reads), lock_at)

    def test_probe_records_no_identify_event(self):
        verify(SUBJECT_MODEL, "alice", "face", sample=b"bob-probe", actor="agent")
        self.assertEqual(_events(ACTION_IDENTIFY), [])

    def test_record_impersonation_suspected_directly(self):
        event = record_impersonation_suspected(
            SUBJECT_MODEL, "alice", modality="face", verification_id="v-1",
            matched_subject_model=SUBJECT_MODEL, matched_subject_id="bob", matched_template_id=str(self.bob.id),
            matched_score=0.9, claimed_score=0.1, threshold=0.62, margin=None, actor="agent",
        )
        self.assertEqual(event.action, ACTION_IMPERSONATION)
        self.assertEqual(event.subject_id, "alice")
        self.assertEqual(event.payload["matched_subject_id"], "bob")
        self.assertEqual(event.payload["matched_score"], 0.9)
        self.assertEqual(event.payload["threshold"], 0.62)


class TestIdentifyEvent(_AuditServiceTestCase):

    def test_actor_records_the_ranking(self):
        enrol(SUBJECT_MODEL, "s1", "face", b"photo-1", actor="agent")
        enrol(SUBJECT_MODEL, "s2", "face", b"photo-2", actor="agent")
        before = BiometricAuditEvent.objects.count()

        matches = identify("face", sample=b"photo-1", top_k=2, actor="investigator")

        events = _events(ACTION_IDENTIFY)
        self.assertEqual(BiometricAuditEvent.objects.count(), before + 1)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual((event.actor, event.subject_id, event.modality), ("investigator", "", "face"))
        self.assertEqual(event.payload["probe"], "sample")
        self.assertEqual(event.payload["top_k"], 2)
        self.assertEqual(
            [(m["subject_id"], m["template_id"]) for m in event.payload["matches"]],
            [(m.subject_id, m.template_id) for m in matches],
        )
        for recorded, match in zip(event.payload["matches"], matches):
            self.assertAlmostEqual(recorded["score"], match.score, places=9)

    def test_vector_and_template_probe_kinds(self):
        identify("face", vector=[0.1] * 8, actor="investigator")
        identify("fingerprint", template=b"tmpl", actor="investigator")
        self.assertEqual([e.payload["probe"] for e in _events(ACTION_IDENTIFY)], ["vector", "template"])
        self.assertNotIn("0.1", json.dumps([e.payload for e in _events(ACTION_IDENTIFY)]))

    def test_no_actor_records_nothing(self):
        identify("face", sample=b"photo-1")
        self.assertEqual(BiometricAuditEvent.objects.count(), 0)

    def test_dedup_scan_records_no_identify_event(self):
        BiometricConfig.dedup_threshold = {"face": 0.0}
        enrol(SUBJECT_MODEL, "s1", "face", b"photo-1", actor="agent")
        enrol(SUBJECT_MODEL, "s2", "face", b"photo-2", actor="agent")

        list(BiometricCandidateSource(modality="face").scan(None))

        self.assertEqual(_events(ACTION_IDENTIFY), [])


class TestConsolidateTemplatesOfPurgeEvents(_AuditServiceTestCase):

    def test_consolidate_records_for_the_kept_subject(self):
        retired = enrol(SUBJECT_MODEL, "retired-1", "face", b"photo", actor="agent")

        counts = consolidate(SUBJECT_MODEL, "kept-1", "retired-1", actor="merger")

        event = _events(ACTION_CONSOLIDATE)[0]
        self.assertEqual((event.subject_id, event.actor), ("kept-1", "merger"))
        self.assertEqual(event.payload, {"retired_id": "retired-1", "counts": counts, "template_ids": [str(retired.id)]})

    def test_consolidate_without_retired_templates_records_nothing(self):
        consolidate(SUBJECT_MODEL, "kept-1", "retired-1", actor="merger")
        self.assertEqual(_events(ACTION_CONSOLIDATE), [])

    def test_templates_of_records_a_read(self):
        template = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent")

        templates_of(SUBJECT_MODEL, "s1", modality="face", actor="reader", purpose="export")

        event = _events(ACTION_TEMPLATE_READ)[0]
        self.assertEqual((event.subject_id, event.actor, event.modality), ("s1", "reader", "face"))
        self.assertEqual(event.payload, {"purpose": "export", "template_ids": [str(template.id)], "modality": "face"})
        self.assertNotIn(json.dumps(template.vector), json.dumps(event.payload))

    def test_purge_records_one_event_per_tombstone(self):
        BiometricRetentionPolicy.objects.create(purge_enabled=True, template_retention_days=30)
        now = timezone.now()
        for subject in ("s1", "s2"):
            stale = enrol(SUBJECT_MODEL, subject, "face", f"old-{subject}".encode(), actor="agent")
            BiometricTemplate.objects.filter(id=stale.id).update(validity_to=now - timedelta(days=40))

        purge(now=now, actor="retention")

        tombstones = {t.subject_id: t for t in BiometricErasure.objects.all()}
        events = _events(ACTION_PURGE)
        self.assertEqual(sorted(e.subject_id for e in events), ["s1", "s2"])
        for event in events:
            tombstone = tombstones[event.subject_id]
            self.assertEqual(event.actor, "retention")
            self.assertEqual(event.payload, {"erasure_id": str(tombstone.id), "reason": "retention", "erased": {"face": 1}})
        self.assertTrue(verify_chain().ok)


class TestDefaultUnchanged(_MultimodalServiceTestCase):
    """The shipped config (audit off) writes no event or alert and takes no lock."""

    def setUp(self):
        restore_audit_settings_on_cleanup(self)
        super().setUp()

    def _scenario(self):
        BiometricRetentionPolicy.objects.create(purge_enabled=True, template_retention_days=30)
        now = timezone.now()
        results = {}
        first = enrol(SUBJECT_MODEL, "s1", "face", b"photo-1", actor="agent")
        enrol(SUBJECT_MODEL, "s2", "face", b"photo-2", actor="agent")
        results["server"] = verify(SUBJECT_MODEL, "s1", "face", sample=b"photo-1", actor="agent")
        results["device"] = verify(SUBJECT_MODEL, "s1", "voice_device", device_score=60.0, actor="agent")
        results["identify"] = [
            (m.subject_id, round(m.score, 9)) for m in identify("face", sample=b"photo-1", actor="investigator")
        ]
        results["templates_of"] = [(t["modality"], t["kind"]) for t in templates_of(SUBJECT_MODEL, "s1", actor="reader")]
        results["consolidate"] = consolidate(SUBJECT_MODEL, "s1", "s2", actor="merger")
        BiometricTemplate.objects.filter(id=first.id).update(validity_to=now - timedelta(days=40))
        tombstone = purge(now=now, actor="retention")
        results["purge"] = (tombstone.subject_id, tombstone.erased)
        results["rows"] = {
            "templates": BiometricTemplate.objects.count(),
            "verifications": BiometricVerification.objects.count(),
            "erasures": BiometricErasure.objects.count(),
        }
        return results

    @staticmethod
    def _comparable(results):
        out = dict(results)
        for key in ("server", "device"):
            r = out[key]
            out[key] = (r.verified, r.confidence, r.origin, r.threshold, r.risk_profile, r.impersonation)
        return out

    def test_no_rows_no_lock_same_results(self):
        self.assertFalse(BiometricConfig.audit.get("enabled"))

        with CaptureQueriesContext(connection) as queries:
            off = self._scenario()

        self.assertEqual(BiometricAuditEvent.objects.count(), 0)
        self.assertEqual(BiometricAlert.objects.count(), 0)
        self.assertFalse([q for q in queries.captured_queries if "pg_advisory_xact_lock" in q["sql"]])
        self.assertFalse([q for q in queries.captured_queries if "biometric_audit_event" in q["sql"]])

        # Same scenario with audit on, on a clean slate: identical results and rows.
        for model in (BiometricVerification, BiometricErasure, BiometricRetentionPolicy, BiometricTemplate):
            model.objects.all().delete()
        from biometric.models import BiometricAccessLog
        BiometricAccessLog.objects.all().delete()
        BiometricConfig.audit = dict(AUDIT_ON)
        on = self._scenario()

        self.assertEqual(self._comparable(off), self._comparable(on))
        self.assertGreater(BiometricAuditEvent.objects.count(), 0)

    def test_refusal_with_audit_off_records_nothing(self):
        self.assertFalse(BiometricConfig.audit.get("enabled"))
        BiometricConfig.quality = {"mode": "enforce", "modalities": {}}
        device = Extracted(vector=[0.1] * 8, face=FaceGeometry(pose={"roll": 30.0}))

        with CaptureQueriesContext(connection) as queries:
            with self.assertRaises(QualityRefusedError):
                enrol(SUBJECT_MODEL, "s1", "face", b"raw", actor="agent", device_template=device)

        self.assertEqual(len(queries.captured_queries), 0)
        self.assertEqual(BiometricAuditEvent.objects.count(), 0)

    def test_services_module_keeps_lazy_imports(self):
        self.assertFalse(hasattr(services, "record_event"))
        self.assertFalse(hasattr(services, "BiometricAuditEvent"))

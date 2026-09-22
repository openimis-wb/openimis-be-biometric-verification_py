"""
Unit tests for services.py's multimodal functions (§3.4, §3.7). Uses fake
providers registered per test — no ML model is ever loaded.
"""

from datetime import timedelta

from cryptography.fernet import Fernet
from django.db import IntegrityError, transaction
from django.test import SimpleTestCase, TestCase
from django.utils import timezone

from biometric.apps import BiometricConfig
from biometric.models import (
    BiometricAccessLog,
    BiometricConsent,
    BiometricErasure,
    BiometricRetentionPolicy,
    BiometricTemplate,
    BiometricVerification,
)
from biometric.providers.base import Extracted
from biometric.providers.device_reported import DeviceReportedMatcher
from biometric.providers.fake import FakeEmbeddingProvider, FakeMatcherProvider
from biometric.registry import ProviderRegistry
from biometric.services import (
    ConsentRequiredError,
    consolidate,
    enrol,
    fuse,
    identify,
    purge,
    templates_of,
    verify,
)
from biometric.signals import on_subject_merged

SUBJECT_MODEL = "individual.Individual"


class _MultimodalServiceTestCase(TestCase):
    """Saves/restores config + provider registry state used by get_provider()."""

    def setUp(self):
        super().setUp()
        self._cfg_snapshot = {
            "modalities": BiometricConfig.modalities,
            "template_key": BiometricConfig.template_key,
            "require_consent": BiometricConfig.require_consent,
            "dedup_threshold": BiometricConfig.dedup_threshold,
            "fusion": BiometricConfig.fusion,
            "vector_index": BiometricConfig.vector_index,
        }
        self._registry_snapshot = dict(ProviderRegistry._modality_registry)
        self._instances_snapshot = dict(ProviderRegistry._modality_instances)
        ProviderRegistry._modality_instances.clear()

        BiometricConfig.modalities = {
            "face": {"provider": "fake_embedding", "threshold": 0.68},
            "fingerprint": {"provider": "fake_matcher", "threshold": 50.0},
            "voice_device": {"provider": "device_reported", "threshold": 48},
        }
        BiometricConfig.template_key = None
        BiometricConfig.require_consent = False
        ProviderRegistry.register_modality("face", "fake_embedding", FakeEmbeddingProvider)
        ProviderRegistry.register_modality("fingerprint", "fake_matcher", FakeMatcherProvider)
        ProviderRegistry.register_modality("voice_device", "device_reported", DeviceReportedMatcher)

    def tearDown(self):
        super().tearDown()
        for key, value in self._cfg_snapshot.items():
            setattr(BiometricConfig, key, value)
        ProviderRegistry._modality_registry.clear()
        ProviderRegistry._modality_registry.update(self._registry_snapshot)
        ProviderRegistry._modality_instances.clear()
        ProviderRegistry._modality_instances.update(self._instances_snapshot)


class TestEnrol(_MultimodalServiceTestCase):

    def test_creates_active_template(self):
        template = enrol(SUBJECT_MODEL, "s1", "face", b"photo-bytes", actor="tester")

        self.assertTrue(template.is_active)
        self.assertEqual(template.subject_model, SUBJECT_MODEL)
        self.assertEqual(template.subject_id, "s1")
        self.assertEqual(template.kind, "embedding")
        self.assertFalse(template.encrypted)
        self.assertIsNotNone(template.vector)

    def test_supersedes_previous_active_row(self):
        first = enrol(SUBJECT_MODEL, "s1", "face", b"photo-1", actor="tester")
        second = enrol(SUBJECT_MODEL, "s1", "face", b"photo-2", actor="tester")

        first.refresh_from_db()
        self.assertIsNotNone(first.validity_to)
        self.assertIsNone(second.validity_to)
        self.assertEqual(
            BiometricTemplate.objects.filter(
                subject_model=SUBJECT_MODEL, subject_id="s1", modality="face", validity_to__isnull=True,
            ).count(),
            1,
        )

    def test_refuses_without_consent_when_required(self):
        BiometricConfig.require_consent = True
        with self.assertRaises(ConsentRequiredError):
            enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="tester")

    def test_allows_when_consent_granted(self):
        BiometricConfig.require_consent = True
        BiometricConsent.objects.create(
            subject_model=SUBJECT_MODEL, subject_id="s1", modality="face",
            granted=True, recorded_by="tester",
        )
        template = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="tester")
        self.assertIsNotNone(template.id)

    def test_device_template_stored_as_given_not_extracted(self):
        device_extracted = Extracted(template=b"device-supplied-template")
        template = enrol(
            SUBJECT_MODEL, "s1", "voice_device", sample=b"unused",
            actor="tester", device_template=device_extracted,
        )
        self.assertEqual(template.template, b"device-supplied-template")
        self.assertEqual(template.kind, "template")

    def test_encrypts_at_rest_when_template_key_set(self):
        key = Fernet.generate_key()
        BiometricConfig.template_key = key

        plaintext_provider = FakeEmbeddingProvider()
        expected_vector = plaintext_provider.extract(b"photo").vector

        template = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="tester")

        self.assertTrue(template.encrypted)
        self.assertNotEqual(template.vector, expected_vector)
        self.assertIsInstance(template.vector, str)


class TestVerify(_MultimodalServiceTestCase):

    def test_server_path_matches_and_records_audit(self):
        enrol(SUBJECT_MODEL, "s1", "face", b"reference-photo", actor="tester")

        result = verify(SUBJECT_MODEL, "s1", "face", sample=b"reference-photo", actor="tester")

        self.assertTrue(result.verified)
        self.assertEqual(result.origin, "server")
        self.assertEqual(result.modality, "face")
        self.assertEqual(
            BiometricVerification.objects.filter(subject_model=SUBJECT_MODEL, subject_id="s1").count(), 1,
        )
        row = BiometricVerification.objects.get(subject_model=SUBJECT_MODEL, subject_id="s1")
        self.assertEqual(row.origin, "server")
        self.assertTrue(row.verified)

    def test_server_path_no_templates_not_verified(self):
        result = verify(SUBJECT_MODEL, "no-templates", "face", sample=b"probe", actor="tester")
        self.assertFalse(result.verified)
        self.assertIsNone(result.confidence)

    def test_device_path_no_extraction_checks_score_against_threshold(self):
        result_pass = verify(
            SUBJECT_MODEL, "s1", "voice_device", device_score=60.0, actor="tester", device_id="tablet-1",
        )
        self.assertTrue(result_pass.verified)
        self.assertEqual(result_pass.origin, "device")

        result_fail = verify(SUBJECT_MODEL, "s1", "voice_device", device_score=10.0, actor="tester")
        self.assertFalse(result_fail.verified)

        self.assertEqual(
            BiometricVerification.objects.filter(subject_model=SUBJECT_MODEL, subject_id="s1", origin="device").count(),
            2,
        )

    def test_keeps_best_score_among_several_templates(self):
        enrol(SUBJECT_MODEL, "s1", "face", b"photo-a", actor="tester", position="left")
        enrol(SUBJECT_MODEL, "s1", "face", b"photo-b", actor="tester", position="right")

        # Probing with photo-a's exact bytes: the "left" template should win with score 1.0.
        result = verify(SUBJECT_MODEL, "s1", "face", sample=b"photo-a", actor="tester")
        self.assertAlmostEqual(result.confidence, 1.0, places=5)


class TestIdentify(_MultimodalServiceTestCase):

    def _make_template(self, subject_id, vector, metadata=None):
        return BiometricTemplate.objects.create(
            subject_model=SUBJECT_MODEL, subject_id=subject_id, modality="face",
            kind="embedding", vector=vector, provider="fake_embedding", model_name="",
            metadata=metadata or {},
        )

    def test_numpy_ranking_orders_by_similarity(self):
        self._make_template("close", [1.0, 0.0, 0.0])
        self._make_template("far", [0.0, 1.0, 0.0])

        matches = identify("face", vector=[1.0, 0.0, 0.0], top_k=5)

        self.assertEqual(matches[0].subject_id, "close")
        self.assertGreater(matches[0].score, matches[1].score)

    def test_scope_filters_on_metadata(self):
        self._make_template("in-scope", [1.0, 0.0], metadata={"cuvee_id": "A"})
        self._make_template("out-of-scope", [1.0, 0.0], metadata={"cuvee_id": "B"})

        matches = identify("face", vector=[1.0, 0.0], scope={"cuvee_id": "A"})

        subject_ids = {m.subject_id for m in matches}
        self.assertIn("in-scope", subject_ids)
        self.assertNotIn("out-of-scope", subject_ids)

    def test_self_exclusion(self):
        self._make_template("self", [1.0, 0.0])
        self._make_template("other", [1.0, 0.0])

        matches = identify("face", vector=[1.0, 0.0], exclude_subject="self")

        subject_ids = {m.subject_id for m in matches}
        self.assertNotIn("self", subject_ids)
        self.assertIn("other", subject_ids)

    def test_top_k_limits_results(self):
        for i in range(5):
            self._make_template(f"s{i}", [1.0, float(i) * 0.01])
        matches = identify("face", vector=[1.0, 0.0], top_k=2)
        self.assertEqual(len(matches), 2)

    def test_pgvector_flag_routes_without_importing_pgvector(self):
        BiometricConfig.vector_index = "pgvector"
        import biometric.services as services_module

        called = {}

        def fake_pgvector_path(provider, modality, probe_vector, top_k, scope, exclude_subject):
            called["hit"] = True
            return []

        original = services_module._identify_pgvector
        services_module._identify_pgvector = fake_pgvector_path
        try:
            result = identify("face", vector=[1.0, 0.0])
        finally:
            services_module._identify_pgvector = original

        self.assertTrue(called.get("hit"))
        self.assertEqual(result, [])

    def test_pgvector_without_app_installed_raises_improperly_configured(self):
        # §6.2: the biometric_pgvector app is the gate, not the pgvector
        # package — ImproperlyConfigured is raised before any import attempt.
        # apps.is_installed is mocked rather than relied on ambiently, so this
        # holds both where biometric_pgvector is absent and where it is
        # actually installed (the pgvector test run — see biometric_pgvector's
        # own tests for the "installed" behaviour).
        from unittest.mock import patch

        from django.apps import apps as django_apps
        from django.core.exceptions import ImproperlyConfigured

        BiometricConfig.vector_index = "pgvector"
        with patch.object(django_apps, "is_installed", return_value=False):
            with self.assertRaises(ImproperlyConfigured):
                identify("face", vector=[1.0, 0.0])


class TestFuse(SimpleTestCase):

    def setUp(self):
        self._modalities = BiometricConfig.modalities
        BiometricConfig.modalities = {
            "face": {"threshold": 0.7},
            "fingerprint": {"threshold": 50.0},
        }

    def tearDown(self):
        BiometricConfig.modalities = self._modalities

    def test_weighted_mean_normalised_to_threshold(self):
        # Both legs exactly at their threshold -> normalised score == 1.0 -> accept.
        decision = fuse(
            {"face": 0.7, "fingerprint": 50.0},
            weights={"face": 1.0, "fingerprint": 1.0},
            thresholds={"accept": 1.0, "review": 0.8},
        )
        self.assertEqual(decision.outcome, "accept")
        self.assertAlmostEqual(decision.score, 1.0, places=5)

    def test_bands_accept_review_reject(self):
        thresholds = {"accept": 1.0, "review": 0.8}
        accept = fuse({"face": 0.7}, weights={"face": 1.0}, thresholds=thresholds)
        review = fuse({"face": 0.6}, weights={"face": 1.0}, thresholds=thresholds)  # 0.6/0.7 ≈ 0.857
        reject = fuse({"face": 0.3}, weights={"face": 1.0}, thresholds=thresholds)  # 0.3/0.7 ≈ 0.43

        self.assertEqual(accept.outcome, "accept")
        self.assertEqual(review.outcome, "review")
        self.assertEqual(reject.outcome, "reject")

    def test_required_leg_missing_caps_at_review(self):
        # Face alone would accept, but fingerprint is required and absent.
        decision = fuse(
            {"face": 0.7, "fingerprint": None},
            weights={"face": 1.0},
            thresholds={"accept": 1.0, "review": 0.5},
            required=frozenset({"fingerprint"}),
        )
        self.assertEqual(decision.outcome, "review")
        self.assertTrue(any("fingerprint" in r for r in decision.reasons))

    def test_floor_breach_forces_floor_decision(self):
        decision = fuse(
            {"face": 0.7, "fingerprint": 5.0},
            weights={"face": 1.0, "fingerprint": 1.0},
            thresholds={"accept": 1.0, "review": 0.5},
            floors={"fingerprint": 20.0},
            floor_decision="reject",
        )
        self.assertEqual(decision.outcome, "reject")

    def test_tighten_only_takes_the_stricter_of_multiple_forced_outcomes(self):
        # required-missing forces >= review; floor breach forces floor_decision=reject.
        # The combined outcome must be reject (the stricter one), never review.
        decision = fuse(
            {"face": 0.7, "voice": None, "fingerprint": 1.0},
            weights={"face": 1.0},
            thresholds={"accept": 1.0, "review": 0.5},
            floors={"fingerprint": 10.0},
            floor_decision="reject",
            required=frozenset({"voice"}),
        )
        self.assertEqual(decision.outcome, "reject")

    def test_no_scored_modality_rejects(self):
        decision = fuse({"face": None}, weights={"face": 1.0})
        self.assertEqual(decision.outcome, "reject")
        self.assertIsNone(decision.score)

    def test_zero_or_negative_weight_leg_ignored(self):
        decision = fuse(
            {"face": 0.0, "fingerprint": 50.0},
            weights={"face": 0.0, "fingerprint": 1.0},
            thresholds={"accept": 1.0, "review": 0.5},
        )
        self.assertAlmostEqual(decision.score, 1.0, places=5)


class TestConsolidate(_MultimodalServiceTestCase):

    def test_repoints_active_templates_to_kept(self):
        enrol(SUBJECT_MODEL, "retired-1", "face", b"photo", actor="tester")

        counts = consolidate(SUBJECT_MODEL, "kept-1", "retired-1", actor="tester")

        self.assertEqual(counts.get("face"), 1)
        self.assertFalse(
            BiometricTemplate.objects.filter(subject_id="retired-1", validity_to__isnull=True).exists()
        )
        self.assertTrue(
            BiometricTemplate.objects.filter(subject_id="kept-1", modality="face", validity_to__isnull=True).exists()
        )

    def test_supersedes_retired_row_on_collision(self):
        enrol(SUBJECT_MODEL, "kept-1", "face", b"kept-photo", actor="tester")
        enrol(SUBJECT_MODEL, "retired-1", "face", b"retired-photo", actor="tester")

        consolidate(SUBJECT_MODEL, "kept-1", "retired-1", actor="tester")

        # kept's original row stays active; retired's row is superseded, not moved.
        kept_active = BiometricTemplate.objects.filter(
            subject_id="kept-1", modality="face", validity_to__isnull=True,
        )
        self.assertEqual(kept_active.count(), 1)
        retired_row = BiometricTemplate.objects.get(subject_id="retired-1", modality="face")
        self.assertIsNotNone(retired_row.validity_to)

    def test_writes_access_log_entry(self):
        enrol(SUBJECT_MODEL, "retired-1", "face", b"photo", actor="tester")
        consolidate(SUBJECT_MODEL, "kept-1", "retired-1", actor="merge-actor")

        log = BiometricAccessLog.objects.get(subject_id="kept-1", purpose="consolidate")
        self.assertEqual(log.actor, "merge-actor")
        self.assertEqual(len(log.template_ids), 1)

    def test_noop_when_retired_has_no_templates(self):
        counts = consolidate(SUBJECT_MODEL, "kept-1", "retired-1", actor="tester")
        self.assertEqual(counts, {})
        self.assertFalse(BiometricAccessLog.objects.filter(subject_id="kept-1").exists())


class TestOnSubjectMergedSignalHandler(_MultimodalServiceTestCase):
    """signals.on_subject_merged bridges deduplication.subject_merged -> consolidate()."""

    def test_flat_kwargs_shape(self):
        enrol(SUBJECT_MODEL, "retired-1", "face", b"photo", actor="tester")

        on_subject_merged(
            sender=None, subject_model=SUBJECT_MODEL,
            kept_id="kept-1", retired_id="retired-1", actor="tester", policy="retire",
        )

        self.assertTrue(
            BiometricTemplate.objects.filter(subject_id="kept-1", validity_to__isnull=True).exists()
        )

    def test_register_service_signal_wrapper_shape(self):
        """Shape produced by core.signals.register_service_signal's AFTER call."""
        enrol(SUBJECT_MODEL, "retired-2", "face", b"photo", actor="tester")

        on_subject_merged(
            sender=None, cls_=None, data=[(), {}], context=None,
            result={
                "subject_model": SUBJECT_MODEL, "kept_id": "kept-2",
                "retired_id": "retired-2", "actor": "tester", "policy": "delete",
            },
        )

        self.assertTrue(
            BiometricTemplate.objects.filter(subject_id="kept-2", validity_to__isnull=True).exists()
        )

    def test_missing_payload_does_not_raise(self):
        on_subject_merged(sender=None, cls_=None, data=None, context=None, result=None)

    def test_bind_service_signals_registers_handler(self):
        from core.signals import REGISTERED_SERVICE_SIGNALS

        registered = REGISTERED_SERVICE_SIGNALS.get("deduplication.subject_merged")
        self.assertIsNotNone(registered)
        if registered.is_signal_registered():
            self.assertIn(on_subject_merged, registered.connected_signals["after"])

    def test_real_deduplication_merge_subjects_triggers_consolidate(self):
        """
        End-to-end through the real deduplication.services.merge_subjects
        (installed side-by-side in this environment): its
        @register_service_signal('deduplication.subject_merged') classmethod
        returns {"subject_model", "kept_id", "retired_id", "actor" (username
        string), "policy"}, delivered to AFTER receivers as kwargs["result"].
        Skips cleanly where deduplication isn't installed.
        """
        try:
            from core.models import User
            from deduplication.services import merge_subjects
            from individual.models import Individual
        except ImportError:
            self.skipTest("deduplication/individual not installed in this environment")

        actor = User.objects.first()
        if actor is None:
            # Bootstrap one: HistoryModel.save() accepts user=self on a brand new
            # (as yet unsaved) instance — the same pattern used to create the very
            # first user in a fresh openIMIS database.
            actor = User(username="dedup-merge-tester")
            actor.save(user=actor)

        from datetime import date

        kept = Individual(first_name="Kept", last_name="Subject", dob=date(1990, 1, 1))
        kept.save(user=actor)
        retired = Individual(first_name="Retired", last_name="Subject", dob=date(1990, 1, 1))
        retired.save(user=actor)

        enrol(SUBJECT_MODEL, str(retired.id), "face", b"photo", actor="tester")

        # merge_subjects emits via transaction.on_commit(), which TestCase's
        # rolled-back transaction never reaches — capture and fire it manually.
        with self.captureOnCommitCallbacks(execute=True):
            merge_subjects(kept, retired, actor, policy="delete")

        self.assertTrue(
            BiometricTemplate.objects.filter(
                subject_model=SUBJECT_MODEL, subject_id=str(kept.id), validity_to__isnull=True,
            ).exists()
        )


class TestPurge(_MultimodalServiceTestCase):

    def test_noop_without_policy_row(self):
        self.assertIsNone(purge())

    def test_noop_when_disabled(self):
        BiometricRetentionPolicy.objects.create(purge_enabled=False, template_retention_days=30)
        self.assertIsNone(purge())

    def test_check_constraint_blocks_enabled_without_days(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                BiometricRetentionPolicy.objects.create(purge_enabled=True, template_retention_days=None)

    def test_purges_stale_templates_and_writes_tombstone(self):
        BiometricRetentionPolicy.objects.create(purge_enabled=True, template_retention_days=30)
        now = timezone.now()

        stale = enrol(SUBJECT_MODEL, "s1", "face", b"old-photo", actor="tester")
        BiometricTemplate.objects.filter(id=stale.id).update(validity_to=now - timedelta(days=40))

        tombstone = purge(now=now, actor="retention")

        self.assertIsNotNone(tombstone)
        self.assertIsInstance(tombstone, BiometricErasure)
        self.assertEqual(tombstone.subject_id, "s1")
        self.assertEqual(tombstone.erased.get("face"), 1)
        self.assertFalse(BiometricTemplate.objects.filter(id=stale.id).exists())

    def test_leaves_active_and_recently_superseded_templates(self):
        BiometricRetentionPolicy.objects.create(purge_enabled=True, template_retention_days=30)
        now = timezone.now()

        active = enrol(SUBJECT_MODEL, "s1", "face", b"current-photo", actor="tester")
        recent = enrol(SUBJECT_MODEL, "s2", "face", b"recent-old-photo", actor="tester")
        BiometricTemplate.objects.filter(id=recent.id).update(validity_to=now - timedelta(days=5))

        result = purge(now=now, actor="retention")

        self.assertIsNone(result)
        self.assertTrue(BiometricTemplate.objects.filter(id=active.id).exists())
        self.assertTrue(BiometricTemplate.objects.filter(id=recent.id).exists())

    # --- active-template retention (§6.3) ---

    def test_active_purge_disabled_by_default_leaves_old_active_template(self):
        BiometricRetentionPolicy.objects.create(
            purge_enabled=False, purge_active_enabled=False,
        )
        now = timezone.now()
        old_active = enrol(SUBJECT_MODEL, "s1", "face", b"old-active-photo", actor="tester")
        BiometricTemplate.objects.filter(id=old_active.id).update(
            validity_from=now - timedelta(days=400)
        )

        result = purge(now=now, actor="retention")

        self.assertIsNone(result)
        self.assertTrue(BiometricTemplate.objects.filter(id=old_active.id).exists())

    def test_active_purge_check_constraint_blocks_enabled_without_days(self):
        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                BiometricRetentionPolicy.objects.create(
                    purge_active_enabled=True, active_template_retention_days=None,
                )

    def test_active_purge_erases_old_active_template_with_active_age_reason(self):
        BiometricRetentionPolicy.objects.create(
            purge_active_enabled=True, active_template_retention_days=365,
        )
        now = timezone.now()
        old_active = enrol(SUBJECT_MODEL, "s1", "face", b"old-active-photo", actor="tester")
        BiometricTemplate.objects.filter(id=old_active.id).update(
            validity_from=now - timedelta(days=400)
        )

        tombstone = purge(now=now, actor="retention")

        self.assertIsNotNone(tombstone)
        self.assertEqual(tombstone.reason, "ACTIVE_AGE")
        self.assertFalse(BiometricTemplate.objects.filter(id=old_active.id).exists())

    def test_active_purge_leaves_recent_active_template(self):
        BiometricRetentionPolicy.objects.create(
            purge_active_enabled=True, active_template_retention_days=365,
        )
        recent_active = enrol(SUBJECT_MODEL, "s1", "face", b"recent-active-photo", actor="tester")

        result = purge(actor="retention")

        self.assertIsNone(result)
        self.assertTrue(BiometricTemplate.objects.filter(id=recent_active.id).exists())

    def test_superseded_purged_before_active(self):
        # Both windows enabled: a superseded row and an old-but-active row are
        # each erased by their own pass, with distinct tombstone reasons.
        BiometricRetentionPolicy.objects.create(
            purge_enabled=True, template_retention_days=30,
            purge_active_enabled=True, active_template_retention_days=365,
        )
        now = timezone.now()

        superseded = enrol(SUBJECT_MODEL, "s1", "face", b"superseded-photo", actor="tester")
        BiometricTemplate.objects.filter(id=superseded.id).update(
            validity_to=now - timedelta(days=40),
        )
        old_active = enrol(SUBJECT_MODEL, "s2", "face", b"old-active-photo", actor="tester")
        BiometricTemplate.objects.filter(id=old_active.id).update(
            validity_from=now - timedelta(days=400),
        )

        purge(now=now, actor="retention")

        self.assertFalse(BiometricTemplate.objects.filter(id=superseded.id).exists())
        self.assertFalse(BiometricTemplate.objects.filter(id=old_active.id).exists())
        reasons = set(
            BiometricErasure.objects.filter(subject_id__in=["s1", "s2"]).values_list("reason", flat=True)
        )
        self.assertEqual(reasons, {"retention", "ACTIVE_AGE"})


class TestTemplatesOf(_MultimodalServiceTestCase):

    def test_decrypts_and_logs_access(self):
        key = Fernet.generate_key()
        BiometricConfig.template_key = key

        plaintext_provider = FakeEmbeddingProvider()
        expected_vector = plaintext_provider.extract(b"photo").vector
        template = enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="tester")

        results = templates_of(SUBJECT_MODEL, "s1", actor="reader", purpose="review")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["vector"], expected_vector)

        log = BiometricAccessLog.objects.get(subject_id="s1", purpose="review")
        self.assertEqual(log.actor, "reader")
        self.assertEqual(log.template_ids, [str(template.id)])

    def test_filters_by_modality(self):
        enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="tester")
        enrol(SUBJECT_MODEL, "s1", "fingerprint", b"print", actor="tester")

        results = templates_of(SUBJECT_MODEL, "s1", modality="face", actor="reader")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["modality"], "face")


class TestSubjectModelDefault(_MultimodalServiceTestCase):
    """subject_model is optional everywhere, defaulting to BIOMETRIC["SUBJECT_MODEL"] (§6.3)."""

    def setUp(self):
        super().setUp()
        self._subject_model_default = BiometricConfig.subject_model
        BiometricConfig.subject_model = "individual.Individual"

    def tearDown(self):
        BiometricConfig.subject_model = self._subject_model_default
        super().tearDown()

    def test_enrol_defaults_subject_model(self):
        template = enrol(subject_id="s1", modality="face", sample=b"photo", actor="tester")
        self.assertEqual(template.subject_model, "individual.Individual")

    def test_verify_defaults_subject_model(self):
        enrol(subject_id="s1", modality="face", sample=b"reference-photo", actor="tester")
        result = verify(subject_id="s1", modality="face", sample=b"reference-photo", actor="tester")
        self.assertTrue(result.verified)
        self.assertTrue(
            BiometricVerification.objects.filter(
                subject_model="individual.Individual", subject_id="s1",
            ).exists()
        )

    def test_consolidate_defaults_subject_model(self):
        enrol(subject_id="retired-1", modality="face", sample=b"photo", actor="tester")
        counts = consolidate(kept_id="kept-1", retired_id="retired-1", actor="tester")
        self.assertEqual(counts.get("face"), 1)
        self.assertTrue(
            BiometricTemplate.objects.filter(
                subject_model="individual.Individual", subject_id="kept-1", validity_to__isnull=True,
            ).exists()
        )

    def test_templates_of_defaults_subject_model(self):
        enrol(subject_id="s1", modality="face", sample=b"photo", actor="tester")
        results = templates_of(subject_id="s1", actor="reader")
        self.assertEqual(len(results), 1)

    def test_explicit_subject_model_still_wins_over_default(self):
        BiometricConfig.subject_model = "individual.Individual"
        template = enrol(
            subject_model="other.Model", subject_id="s1", modality="face",
            sample=b"photo", actor="tester",
        )
        self.assertEqual(template.subject_model, "other.Model")

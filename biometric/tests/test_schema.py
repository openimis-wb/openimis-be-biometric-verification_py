"""
Unit tests for the multimodal GraphQL fields (§3.6): permission guards and
correct delegation to services.py, following the same pattern as the legacy
mutation tests in test_schema.py.
"""

import base64
import io
from types import SimpleNamespace
from unittest import skipUnless
from unittest.mock import MagicMock, patch

import graphene

from django.core.exceptions import PermissionDenied
from django.test import SimpleTestCase

from biometric.apps import BiometricConfig
from biometric.quality import pillow_available
from biometric.risk_profiles import UnknownRiskProfileError
from biometric.schema import (
    BiometricQualityVerdictType,
    EnrolBiometricMutation,
    Mutation,
    Query,
    RecordBiometricConsentMutation,
    VerifyBiometricMutation,
)
from biometric.tests.test_services import _MultimodalServiceTestCase


def _make_info(user):
    info = MagicMock()
    info.context.user = user
    return info


def _anon_user():
    user = MagicMock()
    user.is_anonymous = True
    return user


def _auth_user(username="tester", has_perms=True):
    user = MagicMock()
    user.is_anonymous = False
    user.username = username
    user.has_perms.return_value = has_perms
    return user


class TestEnrolBiometricMutation(SimpleTestCase):

    def test_anonymous_raises_permission_denied(self):
        with self.assertRaises(PermissionDenied):
            EnrolBiometricMutation.mutate(
                None, _make_info(_anon_user()),
                subject_model="individual.Individual", subject_id="s1",
                modality="face", sample=base64.b64encode(b"x").decode(),
            )

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.enrol")
    def test_authenticated_calls_enrol_with_decoded_sample(self, mock_enrol, mock_cfg):
        mock_cfg.gql_biometric_enrol_perms = []
        template = MagicMock()
        template.id = "tid-1"
        template.subject_model = "individual.Individual"
        template.subject_id = "s1"
        template.modality = "face"
        template.position = ""
        template.kind = "embedding"
        template.quality = None
        template.provider = "fake"
        template.model_name = ""
        template.encrypted = False
        template.validity_from = None
        template.validity_to = None
        mock_enrol.return_value = template

        user = _auth_user()
        sample_b64 = base64.b64encode(b"raw-bytes").decode()
        result = EnrolBiometricMutation.mutate(
            None, _make_info(user), subject_model="individual.Individual",
            subject_id="s1", modality="face", sample=sample_b64,
        )

        mock_enrol.assert_called_once()
        args, kwargs = mock_enrol.call_args
        # Positional order forwarded to services.enrol(subject_model, subject_id, modality, sample, ...).
        self.assertEqual(args[0], "individual.Individual")
        self.assertEqual(args[3], b"raw-bytes")
        self.assertEqual(kwargs["actor"], "tester")
        self.assertEqual(result.id, "tid-1")

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.enrol")
    def test_subject_model_is_optional(self, mock_enrol, mock_cfg):
        # §6.3: subject_model is not required — the mutation still accepts
        # the call and forwards None, which services.enrol() defaults.
        mock_cfg.gql_biometric_enrol_perms = []
        template = MagicMock()
        for attr in ("id", "subject_model", "subject_id", "modality", "position", "kind",
                     "quality", "provider", "model_name", "encrypted", "validity_from", "validity_to"):
            setattr(template, attr, None)
        mock_enrol.return_value = template

        EnrolBiometricMutation.mutate(
            None, _make_info(_auth_user()), subject_id="s1",
            modality="face", sample=base64.b64encode(b"x").decode(),
        )

        args, _ = mock_enrol.call_args
        self.assertIsNone(args[0])

    @patch("biometric.schema.BiometricConfig")
    def test_missing_perms_raises_permission_denied(self, mock_cfg):
        mock_cfg.gql_biometric_enrol_perms = ["174001"]
        user = _auth_user(has_perms=False)
        with self.assertRaises(PermissionDenied):
            EnrolBiometricMutation.mutate(
                None, _make_info(user), subject_model="individual.Individual",
                subject_id="s1", modality="face", sample=base64.b64encode(b"x").decode(),
            )

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.enrol")
    def test_data_uri_prefix_is_stripped(self, mock_enrol, mock_cfg):
        mock_cfg.gql_biometric_enrol_perms = []
        template = MagicMock()
        for attr in ("id", "subject_model", "subject_id", "modality", "position", "kind",
                     "quality", "provider", "model_name", "encrypted", "validity_from", "validity_to"):
            setattr(template, attr, None)
        mock_enrol.return_value = template

        b64 = base64.b64encode(b"raw-bytes").decode()
        EnrolBiometricMutation.mutate(
            None, _make_info(_auth_user()), subject_model="individual.Individual",
            subject_id="s1", modality="face", sample=f"data:image/jpeg;base64,{b64}",
        )

        args, _ = mock_enrol.call_args
        self.assertEqual(args[3], b"raw-bytes")


class TestVerifyBiometricMutation(SimpleTestCase):

    def test_anonymous_raises_permission_denied(self):
        with self.assertRaises(PermissionDenied):
            VerifyBiometricMutation.mutate(
                None, _make_info(_anon_user()), subject_model="individual.Individual",
                subject_id="s1", modality="face",
            )

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.verify")
    def test_device_path_forwards_device_score(self, mock_verify, mock_cfg):
        mock_cfg.gql_biometric_verify_perms = []
        result = MagicMock(verified=True, confidence=60.0, provider="device_reported",
                            modality="fingerprint", origin="device", threshold=48.0, error=None)
        mock_verify.return_value = result

        out = VerifyBiometricMutation.mutate(
            None, _make_info(_auth_user()), subject_model="individual.Individual",
            subject_id="s1", modality="fingerprint", device_score=60.0,
        )

        mock_verify.assert_called_once()
        _, kwargs = mock_verify.call_args
        self.assertEqual(kwargs["device_score"], 60.0)
        self.assertIsNone(kwargs["sample"])
        self.assertTrue(out.verified)
        self.assertEqual(out.origin, "device")

    @patch("biometric.schema.BiometricConfig")
    def test_missing_perms_raises_permission_denied(self, mock_cfg):
        mock_cfg.gql_biometric_verify_perms = ["174002"]
        user = _auth_user(has_perms=False)
        with self.assertRaises(PermissionDenied):
            VerifyBiometricMutation.mutate(
                None, _make_info(user), subject_model="individual.Individual",
                subject_id="s1", modality="face",
            )

    @staticmethod
    def _result(risk_profile=""):
        return SimpleNamespace(verified=False, confidence=60.0, provider="device_reported", modality="fingerprint",
                               origin="device", threshold=70.0, error=None, risk_profile=risk_profile)

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.verify")
    def test_forwards_risk_profile(self, mock_verify, mock_cfg):
        mock_cfg.gql_biometric_verify_perms = []
        mock_verify.return_value = self._result("high_risk")

        VerifyBiometricMutation.mutate(
            None, _make_info(_auth_user()), subject_id="s1", modality="fingerprint",
            device_score=60.0, risk_profile="high_risk",
        )

        _, kwargs = mock_verify.call_args
        self.assertEqual(kwargs["risk_profile"], "high_risk")

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.verify")
    def test_omitted_risk_profile_forwards_none(self, mock_verify, mock_cfg):
        mock_cfg.gql_biometric_verify_perms = []
        mock_verify.return_value = self._result()

        out = VerifyBiometricMutation.mutate(
            None, _make_info(_auth_user()), subject_id="s1", modality="fingerprint", device_score=60.0,
        )

        _, kwargs = mock_verify.call_args
        self.assertIsNone(kwargs["risk_profile"])
        self.assertEqual(out.risk_profile, "")

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.verify")
    def test_result_exposes_risk_profile(self, mock_verify, mock_cfg):
        mock_cfg.gql_biometric_verify_perms = []
        mock_verify.return_value = self._result("high_risk")

        out = VerifyBiometricMutation.mutate(
            None, _make_info(_auth_user()), subject_id="s1", modality="fingerprint",
            device_score=60.0, risk_profile="high_risk",
        )

        self.assertEqual(out.risk_profile, "high_risk")
        self.assertEqual(out.threshold, 70.0)
        self.assertFalse(out.verified)

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.verify")
    def test_unknown_risk_profile_error_propagates(self, mock_verify, mock_cfg):
        mock_cfg.gql_biometric_verify_perms = []
        mock_verify.side_effect = UnknownRiskProfileError("Unknown risk profile 'nope'; configured: []")

        with self.assertRaises(UnknownRiskProfileError):
            VerifyBiometricMutation.mutate(
                None, _make_info(_auth_user()), subject_id="s1", modality="fingerprint",
                device_score=60.0, risk_profile="nope",
            )


class TestBiometricVerificationsQuery(SimpleTestCase):

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.models.BiometricVerification")
    def test_returns_risk_profile(self, mock_model, mock_cfg):
        mock_cfg.gql_biometric_read_perms = []
        mock_cfg.subject_model = "individual.Individual"
        row = SimpleNamespace(
            id="v1", subject_model="individual.Individual", subject_id="s1", modality="fingerprint",
            score=60.0, threshold=70.0, verified=False, origin="device", fallback=False, device_id="",
            actor="tester", created_at=None, risk_profile="high_risk",
        )
        mock_model.objects.filter.return_value.order_by.return_value = [row]

        result = Query.resolve_biometric_verifications(None, _make_info(_auth_user()), subject_id="s1")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].risk_profile, "high_risk")
        self.assertEqual(result[0].threshold, 70.0)


class TestVerifyRiskProfileOverGraphQL(_MultimodalServiceTestCase):

    QUERY = (
        'mutation { verifyBiometric(subjectId: "s1", modality: "voice_device", deviceScore: 60, '
        'riskProfile: "%s") { verified threshold riskProfile } }'
    )

    def _execute(self, risk_profile):
        user = MagicMock()
        user.is_anonymous = False
        user.username = "tester"
        user.has_perms.return_value = True
        schema = graphene.Schema(query=Query, mutation=Mutation)
        return schema.execute(self.QUERY % risk_profile, context_value=SimpleNamespace(user=user))

    def test_unknown_profile_is_a_graphql_error_not_verified_false(self):
        BiometricConfig.risk_profiles = {"voice_strict": {"modality_thresholds": {"voice_device": 70}}}

        result = self._execute("nope")

        self.assertIsNone(result.data["verifyBiometric"])
        self.assertIn("Unknown risk profile 'nope'", str(result.errors[0]))


class TestRecordBiometricConsentMutation(SimpleTestCase):

    def test_anonymous_raises_permission_denied(self):
        with self.assertRaises(PermissionDenied):
            RecordBiometricConsentMutation.mutate(
                None, _make_info(_anon_user()), subject_model="individual.Individual",
                subject_id="s1", modality="face", granted=True,
            )

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.models.BiometricConsent")
    def test_creates_consent_row(self, mock_consent_model, mock_cfg):
        mock_cfg.gql_biometric_enrol_perms = []
        created = MagicMock()
        created.id = "consent-1"
        mock_consent_model.objects.create.return_value = created

        result = RecordBiometricConsentMutation.mutate(
            None, _make_info(_auth_user()), subject_model="individual.Individual",
            subject_id="s1", modality="face", granted=True, note="phone call",
        )

        mock_consent_model.objects.create.assert_called_once_with(
            subject_model="individual.Individual", subject_id="s1", modality="face",
            granted=True, recorded_by="tester", note="phone call",
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.id, "consent-1")


class TestIdentifyBiometricQuery(SimpleTestCase):

    def test_anonymous_raises_permission_denied(self):
        with self.assertRaises(PermissionDenied):
            Query.resolve_identify_biometric(
                None, _make_info(_anon_user()), modality="face", sample=base64.b64encode(b"x").decode(),
            )

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.identify")
    def test_authenticated_returns_matches(self, mock_identify, mock_cfg):
        mock_cfg.gql_biometric_identify_perms = []
        match = MagicMock(subject_model="individual.Individual", subject_id="s2", template_id="t2", score=0.9)
        mock_identify.return_value = [match]

        result = Query.resolve_identify_biometric(
            None, _make_info(_auth_user()), modality="face", sample=base64.b64encode(b"probe").decode(),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].subject_id, "s2")
        mock_identify.assert_called_once()
        _, kwargs = mock_identify.call_args
        self.assertEqual(kwargs["sample"], b"probe")


class TestBiometricTemplatesQuery(SimpleTestCase):

    def test_anonymous_raises_permission_denied(self):
        with self.assertRaises(PermissionDenied):
            Query.resolve_biometric_templates(
                None, _make_info(_anon_user()), subject_model="individual.Individual", subject_id="s1",
            )

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.models.BiometricTemplate")
    def test_authenticated_returns_template_metadata_only(self, mock_model, mock_cfg):
        mock_cfg.gql_biometric_read_perms = []
        row = MagicMock()
        row.id = "t1"
        row.subject_model = "individual.Individual"
        row.subject_id = "s1"
        row.modality = "face"
        row.position = ""
        row.kind = "embedding"
        row.quality = 88.0
        row.provider = "deepface"
        row.model_name = "ArcFace"
        row.encrypted = True
        row.validity_from = None
        row.validity_to = None
        mock_model.objects.filter.return_value = [row]

        result = Query.resolve_biometric_templates(
            None, _make_info(_auth_user()), subject_model="individual.Individual", subject_id="s1",
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].modality, "face")
        self.assertFalse(hasattr(result[0], "vector"))


VERDICT = {
    "status": "REFUSED",
    "mode": "advisory",
    "modality": "face",
    "reasons": ["sharpness_below_min"],
    "measures": [
        {"name": "sharpness", "value": 3.0, "limit": 100.0, "kind": "min", "passed": False,
         "source": "image", "detail": ""},
        {"name": "pitch", "value": -30.0, "limit": None, "kind": "max", "passed": None,
         "source": "provider_pose", "detail": ""},
    ],
    "version": 1,
}


def _template_row(quality_verdict):
    row = MagicMock()
    row.id = "t1"
    row.subject_model = "individual.Individual"
    row.subject_id = "s1"
    row.modality = "face"
    row.position = ""
    row.kind = "embedding"
    row.quality = None
    row.provider = "fake"
    row.model_name = ""
    row.encrypted = False
    row.validity_from = None
    row.validity_to = None
    if quality_verdict is not MagicMock:
        row.quality_verdict = quality_verdict
    return row


class TestQualityVerdictField(SimpleTestCase):

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.models.BiometricTemplate")
    def test_templates_query_maps_stored_verdict(self, mock_model, mock_cfg):
        mock_cfg.gql_biometric_read_perms = []
        mock_model.objects.filter.return_value = [_template_row(VERDICT)]

        result = Query.resolve_biometric_templates(None, _make_info(_auth_user()), subject_id="s1")

        verdict = result[0].quality_verdict
        self.assertIsInstance(verdict, BiometricQualityVerdictType)
        self.assertEqual(verdict.status, "REFUSED")
        self.assertEqual(verdict.reasons, ["sharpness_below_min"])
        self.assertFalse(verdict.measures[0].passed)
        self.assertIsNone(verdict.measures[1].passed)
        self.assertEqual(verdict.measures[1].value, -30.0)

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.models.BiometricTemplate")
    def test_templates_query_non_dict_verdict_is_null(self, mock_model, mock_cfg):
        mock_cfg.gql_biometric_read_perms = []
        mock_model.objects.filter.return_value = [_template_row(MagicMock), _template_row(None)]

        result = Query.resolve_biometric_templates(None, _make_info(_auth_user()), subject_id="s1")

        self.assertIsNone(result[0].quality_verdict)
        self.assertIsNone(result[1].quality_verdict)

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.enrol")
    def test_enrol_mutation_returns_row_verdict(self, mock_enrol, mock_cfg):
        mock_cfg.gql_biometric_enrol_perms = []
        mock_enrol.return_value = _template_row(VERDICT)

        result = EnrolBiometricMutation.mutate(
            None, _make_info(_auth_user()), subject_id="s1", modality="face",
            sample=base64.b64encode(b"x").decode(),
        )

        self.assertEqual(result.quality_verdict.status, "REFUSED")
        self.assertEqual(result.quality_verdict.mode, "advisory")


@skipUnless(pillow_available(), "Pillow is not importable")
class TestEnrolQualityRefusalOverGraphQL(_MultimodalServiceTestCase):

    QUERY = (
        'mutation { enrolBiometric(subjectId: "s1", modality: "face", sample: "%s") '
        "{ id qualityVerdict { status reasons } } }"
    )

    def _execute(self):
        import numpy as np
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(buffer, format="PNG")
        sample = base64.b64encode(buffer.getvalue()).decode()
        user = MagicMock()
        user.is_anonymous = False
        user.username = "tester"
        user.has_perms.return_value = True
        schema = graphene.Schema(query=Query, mutation=Mutation)
        return schema.execute(self.QUERY % sample, context_value=SimpleNamespace(user=user))

    def test_enforce_refusal_is_a_graphql_error_with_extensions(self):
        BiometricConfig.quality = {"mode": "enforce"}

        result = self._execute()

        self.assertIsNone(result.data["enrolBiometric"])
        error = result.errors[0]
        self.assertEqual(error.extensions["code"], "BIOMETRIC_QUALITY_REFUSED")
        self.assertEqual(error.extensions["verdict"]["reasons"], ["sharpness_below_min"])
        self.assertEqual(error.extensions["verdict"]["status"], "REFUSED")
        self.assertNotIn("s1", str(error))

    def test_advisory_returns_the_verdict_with_the_row(self):
        BiometricConfig.quality = {"mode": "advisory"}

        result = self._execute()

        self.assertIsNone(result.errors)
        payload = result.data["enrolBiometric"]
        self.assertIsNotNone(payload["id"])
        self.assertEqual(payload["qualityVerdict"]["status"], "REFUSED")
        self.assertEqual(payload["qualityVerdict"]["reasons"], ["sharpness_below_min"])


def _probe(**overrides):
    from biometric.impersonation import ImpersonationProbe

    candidate = {"subject_model": "individual.Individual", "subject_id": "bob", "template_id": "t-bob",
                 "score": 0.99, "suspect": True}
    values = dict(status="ok", suspected=True, threshold=0.62, margin=None, top_k=5, claimed_score=0.05,
                  best_match=candidate, candidates=[candidate], latency_ms=3.5)
    values.update(overrides)
    return ImpersonationProbe(**values)


def _verify_only_user():
    user = _auth_user()
    user.has_perms.side_effect = lambda perms: "174003" not in perms
    return user


class TestImpersonationOverGraphQL(SimpleTestCase):

    @staticmethod
    def _cfg(mock_cfg):
        mock_cfg.gql_biometric_verify_perms = ["174002"]
        mock_cfg.gql_biometric_identify_perms = ["174003"]
        mock_cfg.gql_biometric_read_perms = ["174004"]
        mock_cfg.subject_model = "individual.Individual"

    @staticmethod
    def _result(impersonation):
        from biometric.providers.base import VerificationResult

        return VerificationResult(verified=False, confidence=0.05, provider="deepface", modality="face",
                                  origin="server", threshold=0.32, impersonation=impersonation)

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.verify")
    def test_verify_mutation_exposes_the_probe_with_identify_perms(self, mock_verify, mock_cfg):
        self._cfg(mock_cfg)
        mock_verify.return_value = self._result(_probe())

        out = VerifyBiometricMutation.mutate(
            None, _make_info(_auth_user()), subject_id="alice", modality="face", sample="YWJj",
        )

        self.assertEqual(out.impersonation.status, "ok")
        self.assertTrue(out.impersonation.suspected)
        self.assertEqual(out.impersonation.matched_subject_model, "individual.Individual")
        self.assertEqual(out.impersonation.matched_subject_id, "bob")
        self.assertEqual(out.impersonation.matched_score, 0.99)
        self.assertEqual(out.impersonation.claimed_score, 0.05)
        self.assertEqual(out.impersonation.threshold, 0.62)
        self.assertEqual([c.subject_id for c in out.impersonation.candidates], ["bob"])
        self.assertEqual(out.impersonation.candidates[0].template_id, "t-bob")

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.verify")
    def test_verify_mutation_hides_identities_without_identify_perms(self, mock_verify, mock_cfg):
        self._cfg(mock_cfg)
        mock_verify.return_value = self._result(_probe())

        out = VerifyBiometricMutation.mutate(
            None, _make_info(_verify_only_user()), subject_id="alice", modality="face", sample="YWJj",
        )

        self.assertEqual(out.impersonation.status, "ok")
        self.assertTrue(out.impersonation.suspected)
        self.assertEqual(out.impersonation.matched_score, 0.99)
        self.assertIsNone(out.impersonation.matched_subject_model)
        self.assertIsNone(out.impersonation.matched_subject_id)
        self.assertEqual(out.impersonation.candidates, [])

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.services.verify")
    def test_verify_mutation_without_probe_is_null(self, mock_verify, mock_cfg):
        self._cfg(mock_cfg)
        mock_verify.return_value = self._result(None)

        out = VerifyBiometricMutation.mutate(
            None, _make_info(_auth_user()), subject_id="alice", modality="face", sample="YWJj",
        )

        self.assertIsNone(out.impersonation)

    @staticmethod
    def _row(**overrides):
        values = dict(
            id="v1", subject_model="individual.Individual", subject_id="alice", modality="face",
            score=0.05, threshold=0.32, verified=False, origin="server", fallback=False, device_id="",
            actor="tester", created_at=None, risk_profile="",
            impersonation_status="ok", impersonation_suspected=True,
            impersonation_subject_model="individual.Individual", impersonation_subject_id="bob",
            impersonation_score=0.99, impersonation_evidence=_probe().as_evidence(),
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.models.BiometricVerification")
    def test_verifications_query_builds_the_probe_from_the_row(self, mock_model, mock_cfg):
        self._cfg(mock_cfg)
        mock_model.objects.filter.return_value.order_by.return_value = [self._row()]

        result = Query.resolve_biometric_verifications(None, _make_info(_auth_user()), subject_id="alice")

        probe = result[0].impersonation
        self.assertEqual(probe.status, "ok")
        self.assertTrue(probe.suspected)
        self.assertEqual(probe.matched_subject_id, "bob")
        self.assertEqual(probe.matched_score, 0.99)
        self.assertEqual(probe.claimed_score, 0.05)
        self.assertEqual(probe.latency_ms, 3.5)
        self.assertEqual([c.subject_id for c in probe.candidates], ["bob"])

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.models.BiometricVerification")
    def test_verifications_query_hides_identities_without_identify_perms(self, mock_model, mock_cfg):
        self._cfg(mock_cfg)
        mock_model.objects.filter.return_value.order_by.return_value = [self._row()]

        result = Query.resolve_biometric_verifications(None, _make_info(_verify_only_user()), subject_id="alice")

        probe = result[0].impersonation
        self.assertTrue(probe.suspected)
        self.assertEqual(probe.matched_score, 0.99)
        self.assertIsNone(probe.matched_subject_id)
        self.assertIsNone(probe.matched_subject_model)
        self.assertEqual(probe.candidates, [])

    @patch("biometric.schema.BiometricConfig")
    @patch("biometric.models.BiometricVerification")
    def test_verifications_query_row_without_probe_is_null(self, mock_model, mock_cfg):
        self._cfg(mock_cfg)
        mock_model.objects.filter.return_value.order_by.return_value = [
            self._row(impersonation_status="", impersonation_suspected=False, impersonation_subject_model="",
                      impersonation_subject_id="", impersonation_score=None, impersonation_evidence={}),
        ]

        result = Query.resolve_biometric_verifications(None, _make_info(_auth_user()), subject_id="alice")

        self.assertIsNone(result[0].impersonation)

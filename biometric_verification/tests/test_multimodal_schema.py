"""
Unit tests for the multimodal GraphQL fields (§3.6): permission guards and
correct delegation to services.py, following the same pattern as the legacy
mutation tests in test_schema.py.
"""

import base64
from unittest.mock import MagicMock, patch

from django.core.exceptions import PermissionDenied
from django.test import SimpleTestCase

from biometric_verification.schema import (
    EnrolBiometricMutation,
    Query,
    RecordBiometricConsentMutation,
    VerifyBiometricMutation,
)


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

    @patch("biometric_verification.schema.BiometricVerificationConfig")
    @patch("biometric_verification.services.enrol")
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
        self.assertEqual(args[3], b"raw-bytes")
        self.assertEqual(kwargs["actor"], "tester")
        self.assertEqual(result.id, "tid-1")

    @patch("biometric_verification.schema.BiometricVerificationConfig")
    def test_missing_perms_raises_permission_denied(self, mock_cfg):
        mock_cfg.gql_biometric_enrol_perms = ["174001"]
        user = _auth_user(has_perms=False)
        with self.assertRaises(PermissionDenied):
            EnrolBiometricMutation.mutate(
                None, _make_info(user), subject_model="individual.Individual",
                subject_id="s1", modality="face", sample=base64.b64encode(b"x").decode(),
            )

    @patch("biometric_verification.schema.BiometricVerificationConfig")
    @patch("biometric_verification.services.enrol")
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

    @patch("biometric_verification.schema.BiometricVerificationConfig")
    @patch("biometric_verification.services.verify")
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

    @patch("biometric_verification.schema.BiometricVerificationConfig")
    def test_missing_perms_raises_permission_denied(self, mock_cfg):
        mock_cfg.gql_biometric_verify_perms = ["174002"]
        user = _auth_user(has_perms=False)
        with self.assertRaises(PermissionDenied):
            VerifyBiometricMutation.mutate(
                None, _make_info(user), subject_model="individual.Individual",
                subject_id="s1", modality="face",
            )


class TestRecordBiometricConsentMutation(SimpleTestCase):

    def test_anonymous_raises_permission_denied(self):
        with self.assertRaises(PermissionDenied):
            RecordBiometricConsentMutation.mutate(
                None, _make_info(_anon_user()), subject_model="individual.Individual",
                subject_id="s1", modality="face", granted=True,
            )

    @patch("biometric_verification.schema.BiometricVerificationConfig")
    @patch("biometric_verification.models.BiometricConsent")
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

    @patch("biometric_verification.schema.BiometricVerificationConfig")
    @patch("biometric_verification.services.identify")
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

    @patch("biometric_verification.schema.BiometricVerificationConfig")
    @patch("biometric_verification.models.BiometricTemplate")
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

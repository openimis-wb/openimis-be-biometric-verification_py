"""
Impersonation probe on the device-reported path of verify()
(docs/wb-biometric-dedup-seam.md §6.9).

With a device score there is no server-side sample. The probe runs only when
the device also supplies what it extracted (a vector for an embedding
modality, a template for a template modality), the modality's provider can
rank the gallery on the server, and IMPERSONATION_PROBE["device_path"] is
true. Otherwise an enabled probe records why it did not run.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from biometric import services
from biometric.apps import BiometricConfig
from biometric.audit_chain import ACTION_VERIFY
from biometric.impersonation import (
    PROBE_DEFAULTS,
    SKIP_DEVICE_PATH_DISABLED,
    SKIP_NO_DEVICE_TEMPLATE,
    SKIP_PROVIDER_MATCHES_ON_DEVICE,
    SKIP_REASONS,
)
from biometric.models import BiometricAuditEvent, BiometricVerification
from biometric.providers.base import Extracted
from biometric.schema import Query, VerifyBiometricMutation
from biometric.services import verify, verify_multimodal
from biometric.tests.test_audit_chain import restore_audit_settings_on_cleanup
from biometric.tests.test_impersonation import VECTORS, _ImpersonationTestCase
from biometric.tests.test_services import SUBJECT_MODEL


class TestDevicePathProbe(_ImpersonationTestCase):

    def setUp(self):
        super().setUp()
        self._face("alice", VECTORS[b"alice"])
        self.bob = self._face("bob", VECTORS[b"bob"])
        self.device_face = Extracted(vector=list(VECTORS[b"bob-probe"]))

    def test_skip_reasons_constant(self):
        self.assertEqual(
            SKIP_REASONS, ("provider_matches_on_device", "no_device_template", "device_path_disabled"),
        )
        self.assertIs(PROBE_DEFAULTS["device_path"], False)
        from biometric.apps import DEFAULT_CFG

        self.assertIs(DEFAULT_CFG["impersonation_probe"]["device_path"], False)

    def test_probe_off_records_no_reason(self):
        with patch("biometric.services.identify") as identify:
            result = verify(SUBJECT_MODEL, "alice", "face", device_score=0.9, device_template=self.device_face,
                            actor="tester")

        identify.assert_not_called()
        self.assertIsNone(result.impersonation)
        self.assertEqual(result.impersonation_skip_reason, "")
        self.assertEqual(self._row().impersonation_skip_reason, "")

    def test_device_path_disabled(self):
        self._enable()

        with patch("biometric.services.identify") as identify:
            with self.assertNumQueries(1):
                result = verify(SUBJECT_MODEL, "alice", "face", device_score=0.9,
                                device_template=self.device_face, actor="tester")

        identify.assert_not_called()
        self.assertIsNone(result.impersonation)
        self.assertEqual(result.impersonation_skip_reason, SKIP_DEVICE_PATH_DISABLED)
        row = self._row()
        self.assertEqual((row.impersonation_status, row.impersonation_skip_reason), ("", SKIP_DEVICE_PATH_DISABLED))

    def test_no_device_template(self):
        self._enable(device_path=True)
        cases = {"nothing supplied": None, "template bytes for an embedding modality": Extracted(template=b"x")}
        for name, device_template in cases.items():
            with self.subTest(name):
                with patch("biometric.services.identify") as identify:
                    result = verify(SUBJECT_MODEL, "alice", "face", device_score=0.9,
                                    device_template=device_template, actor="tester")
                identify.assert_not_called()
                self.assertEqual(result.impersonation_skip_reason, SKIP_NO_DEVICE_TEMPLATE)
                self.assertEqual(self._row().impersonation_skip_reason, SKIP_NO_DEVICE_TEMPLATE)

    def test_device_reported_provider_cannot_rank(self):
        self._enable(modalities=["voice_device"], device_path=True)
        self._template("bob", "voice_device", "device_reported", b"voice-bob")

        with patch("biometric.services.identify") as identify:
            result = verify(SUBJECT_MODEL, "alice", "voice_device", device_score=60.0,
                            device_template=Extracted(template=b"voice-bob"), actor="tester")

        identify.assert_not_called()
        self.assertIsNone(result.impersonation)
        self.assertEqual(result.impersonation_skip_reason, SKIP_PROVIDER_MATCHES_ON_DEVICE)
        self.assertEqual(self._row().impersonation_skip_reason, SKIP_PROVIDER_MATCHES_ON_DEVICE)

    def test_opt_in_probe_ranks_the_device_vector(self):
        self._enable(device_path=True)

        with patch("biometric.services.identify", wraps=services.identify) as identify:
            result = verify(SUBJECT_MODEL, "alice", "face", device_score=0.9, device_template=self.device_face,
                            actor="tester")

        _, kwargs = identify.call_args
        self.assertNotIn("sample", kwargs)
        self.assertEqual(kwargs["vector"], VECTORS[b"bob-probe"])
        self.assertEqual((result.origin, result.confidence, result.verified), ("device", 0.9, True))
        self.assertEqual(result.threshold, 0.68)
        self.assertEqual(result.impersonation.status, "ok")
        self.assertTrue(result.impersonation.suspected)
        self.assertEqual(result.impersonation.best_match["subject_id"], "bob")
        self.assertEqual(result.impersonation_skip_reason, "")
        row = self._row()
        self.assertEqual((row.origin, row.score, row.impersonation_status), ("device", 0.9, "ok"))
        self.assertTrue(row.impersonation_suspected)
        self.assertEqual(row.impersonation_subject_id, "bob")
        self.assertEqual(row.impersonation_skip_reason, "")

    def test_opt_in_probe_on_a_template_modality(self):
        self._enable(modalities=["fingerprint"], device_path=True)
        self._template("alice", "fingerprint", "fake_matcher", b"print-alice")
        self._template("bob", "fingerprint", "fake_matcher", b"print-bob")

        result = verify(SUBJECT_MODEL, "alice", "fingerprint", device_score=70.0,
                        device_template=Extracted(template=b"print-bob"), actor="tester")

        self.assertTrue(result.verified)
        self.assertTrue(result.impersonation.suspected)
        self.assertEqual(result.impersonation.best_match["subject_id"], "bob")
        self.assertEqual(result.impersonation.claimed_score, 0.0)

    def test_suspicion_on_the_device_path_emits_the_signal(self):
        self._enable(device_path=True)
        fired = []
        self._bind_receiver(lambda **kwargs: fired.append(kwargs))

        with self.captureOnCommitCallbacks(execute=True):
            verify(SUBJECT_MODEL, "alice", "face", device_score=0.9, device_template=self.device_face,
                   actor="tester", device_id="tab-3")

        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0]["result"]["matched_subject_id"], "bob")
        self.assertEqual(fired[0]["result"]["device_id"], "tab-3")

    def test_device_template_on_the_server_path_is_refused_before_any_row(self):
        with self.assertRaises(ValueError):
            verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", device_template=self.device_face,
                   actor="tester")
        self.assertEqual(BiometricVerification.objects.count(), 0)

    def test_multimodal_device_leg_carries_its_template(self):
        self._enable(device_path=True)

        result = verify_multimodal(
            SUBJECT_MODEL, "alice", [{"modality": "face", "device_score": 0.9, "device_template": self.device_face}],
            actor="tester",
        )

        self.assertTrue(result.legs[0].impersonation.suspected)

    def test_audit_payload_names_the_reason_and_never_the_vector(self):
        restore_audit_settings_on_cleanup(self)
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        self._enable()
        vector = [0.123456 + i for i in range(20)]

        verify(SUBJECT_MODEL, "alice", "face", device_score=0.9, device_template=Extracted(vector=vector),
               actor="tester")

        event = BiometricAuditEvent.objects.get(action=ACTION_VERIFY)
        self.assertEqual(event.payload["impersonation_skip_reason"], SKIP_DEVICE_PATH_DISABLED)
        self.assertNotIn("0.123456", json.dumps(event.payload))


def _user():
    user = MagicMock()
    user.is_anonymous = False
    user.username = "tester"
    user.has_perms.return_value = True
    return user


def _info(user):
    info = MagicMock()
    info.context.user = user
    return info


class TestDevicePathOverGraphQL(_ImpersonationTestCase):

    @patch("biometric.services.verify")
    def test_device_vector_and_template_are_forwarded(self, mock_verify):
        from biometric.providers.base import VerificationResult

        mock_verify.return_value = VerificationResult(
            verified=True, confidence=0.9, modality="face", origin="device", threshold=0.68,
            impersonation_skip_reason=SKIP_DEVICE_PATH_DISABLED,
        )

        out = VerifyBiometricMutation.mutate(
            None, _info(_user()), subject_id="alice", modality="face", device_score=0.9,
            device_vector=[0.1, 0.2], device_template="dGVtcGxhdGU=",
        )

        _, kwargs = mock_verify.call_args
        self.assertEqual(kwargs["device_template"], Extracted(vector=[0.1, 0.2], template=b"template"))
        self.assertEqual(out.impersonation_skip_reason, SKIP_DEVICE_PATH_DISABLED)

    @patch("biometric.services.verify")
    def test_no_device_probe_input_forwards_none(self, mock_verify):
        from biometric.providers.base import VerificationResult

        mock_verify.return_value = VerificationResult(verified=True, confidence=0.9, origin="device")

        out = VerifyBiometricMutation.mutate(
            None, _info(_user()), subject_id="alice", modality="face", device_score=0.9,
        )

        _, kwargs = mock_verify.call_args
        self.assertIsNone(kwargs["device_template"])
        self.assertEqual(out.impersonation_skip_reason, "")

    def test_end_to_end_and_the_verifications_query(self):
        import graphene

        from biometric.schema import Mutation

        self._face("alice", VECTORS[b"alice"])
        self._face("bob", VECTORS[b"bob"])
        self._enable(device_path=True)
        schema = graphene.Schema(query=Query, mutation=Mutation)
        query = (
            'mutation { verifyBiometric(subjectId: "alice", modality: "face", deviceScore: 0.9, '
            "deviceVector: [0.05, 1.0, 0.0]) { verified impersonationSkipReason impersonation { status suspected } } }"
        )

        result = schema.execute(query, context_value=SimpleNamespace(user=_user()))

        self.assertIsNone(result.errors, result.errors)
        data = result.data["verifyBiometric"]
        self.assertEqual(data["impersonation"], {"status": "ok", "suspected": True})
        self.assertEqual(data["impersonationSkipReason"], "")

        self._enable()
        schema.execute(query, context_value=SimpleNamespace(user=_user()))
        rows = Query.resolve_biometric_verifications(None, _info(_user()), subject_id="alice")
        self.assertEqual([r.impersonation_skip_reason for r in rows], [SKIP_DEVICE_PATH_DISABLED, ""])

    def test_multimodal_leg_device_vector_over_graphql(self):
        import graphene

        from biometric.schema import Mutation

        self._face("alice", VECTORS[b"alice"])
        self._face("bob", VECTORS[b"bob"])
        self._enable(device_path=True)
        query = (
            'mutation { verifyBiometricMultimodal(subjectId: "alice", legs: [{modality: "face", deviceScore: 0.9, '
            "deviceVector: [0.05, 1.0, 0.0]}]) { outcome legs { impersonationSkipReason impersonation { suspected } } } }"
        )

        result = graphene.Schema(query=Query, mutation=Mutation).execute(
            query, context_value=SimpleNamespace(user=_user()),
        )

        self.assertIsNone(result.errors, result.errors)
        leg = result.data["verifyBiometricMultimodal"]["legs"][0]
        self.assertEqual(leg, {"impersonationSkipReason": "", "impersonation": {"suspected": True}})

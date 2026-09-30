"""
Impersonation probe on the device-reported path of verify()
(docs/wb-biometric-dedup-seam.md §6.9).

With a device score there is no server-side sample. verify() accepts a
device score only for a device_reported provider, which cannot rank the
gallery on the server, so an enabled probe records provider_matches_on_device
there. A server-matched modality refuses the device score before the probe.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from biometric.apps import BiometricConfig
from biometric.audit_chain import ACTION_VERIFY
from biometric.impersonation import (
    PROBE_DEFAULTS,
    SKIP_DEVICE_PATH_DISABLED,
    SKIP_NO_DEVICE_TEMPLATE,
    SKIP_PROVIDER_MATCHES_ON_DEVICE,
    SKIP_REASONS,
    device_path_probe,
)
from biometric.models import BiometricAuditEvent, BiometricVerification
from biometric.providers.base import Extracted
from biometric.schema import Query, VerifyBiometricMutation
from biometric.registry import ProviderRegistry
from biometric.services import DevicePathRefusedError, verify, verify_multimodal
from biometric.tests.test_audit_chain import restore_audit_settings_on_cleanup
from biometric.tests.test_impersonation import VECTORS, _ImpersonationTestCase
from biometric.tests.test_services import SUBJECT_MODEL
from biometric.tests.synthetic_subjects import SyntheticSubjectsMixin


class TestDevicePathProbe(_ImpersonationTestCase):

    def setUp(self):
        super().setUp()
        self._face("alice", VECTORS[b"alice"])
        self.bob = self._face("bob", VECTORS[b"bob"])
        self.device_face = Extracted(vector=list(VECTORS[b"bob-probe"]))
        self._template("bob", "voice_device", "device_reported", b"voice-bob")

    def test_skip_reasons_constant(self):
        self.assertEqual(
            SKIP_REASONS, ("provider_matches_on_device", "no_device_template", "device_path_disabled"),
        )
        self.assertIs(PROBE_DEFAULTS["device_path"], False)
        from biometric.apps import DEFAULT_CFG

        self.assertIs(DEFAULT_CFG["impersonation_probe"]["device_path"], False)

    def test_probe_off_records_no_reason(self):
        with patch("biometric.services.identify") as identify:
            result = verify(SUBJECT_MODEL, "alice", "voice_device", device_score=60.0,
                            device_template=Extracted(template=b"voice-bob"), actor="tester")

        identify.assert_not_called()
        self.assertIsNone(result.impersonation)
        self.assertEqual(result.impersonation_skip_reason, "")
        self.assertEqual(self._row().impersonation_skip_reason, "")

    def test_device_reported_provider_cannot_rank(self):
        self._enable(modalities=["voice_device"], device_path=True)

        with patch("biometric.services.identify") as identify:
            result = verify(SUBJECT_MODEL, "alice", "voice_device", device_score=60.0,
                            device_template=Extracted(template=b"voice-bob"), actor="tester")

        identify.assert_not_called()
        self.assertIsNone(result.impersonation)
        self.assertEqual(result.impersonation_skip_reason, SKIP_PROVIDER_MATCHES_ON_DEVICE)
        self.assertEqual(self._row().impersonation_skip_reason, SKIP_PROVIDER_MATCHES_ON_DEVICE)

    def test_server_matched_modalities_refuse_the_device_path_before_the_probe(self):
        self._template("alice", "fingerprint", "fake_matcher", b"print-alice")
        self._template("bob", "fingerprint", "fake_matcher", b"print-bob")
        cases = {
            "face": {"device_score": 0.9, "device_template": self.device_face},
            "fingerprint": {"device_score": 70.0, "device_template": Extracted(template=b"print-bob")},
        }
        for device_path in (False, True):
            self._enable(modalities=["face", "fingerprint"], device_path=device_path)
            for modality, kwargs in cases.items():
                with self.subTest(modality=modality, device_path=device_path):
                    with patch("biometric.services.identify") as identify:
                        with self.assertRaises(DevicePathRefusedError):
                            verify(SUBJECT_MODEL, "alice", modality, actor="tester", **kwargs)
                    identify.assert_not_called()
        self.assertEqual(BiometricVerification.objects.count(), 0)

    def test_multimodal_device_leg_is_refused_before_any_row(self):
        self._enable(device_path=True)

        with self.assertRaises(DevicePathRefusedError):
            verify_multimodal(
                SUBJECT_MODEL, "alice",
                [{"modality": "face", "device_score": 0.9, "device_template": self.device_face}],
                actor="tester",
            )
        self.assertEqual(BiometricVerification.objects.count(), 0)

    def test_device_template_on_the_server_path_is_refused_before_any_row(self):
        with self.assertRaises(ValueError):
            verify(SUBJECT_MODEL, "alice", "face", sample=b"alice", device_template=self.device_face,
                   actor="tester")
        self.assertEqual(BiometricVerification.objects.count(), 0)

    def test_audit_payload_names_the_reason_and_never_the_template(self):
        restore_audit_settings_on_cleanup(self)
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        self._enable(modalities=["voice_device"])

        verify(SUBJECT_MODEL, "alice", "voice_device", device_score=60.0,
               device_template=Extracted(template=b"voice-template-SENTINEL"), actor="tester")

        event = BiometricAuditEvent.objects.get(action=ACTION_VERIFY)
        self.assertEqual(event.payload["impersonation_skip_reason"], SKIP_PROVIDER_MATCHES_ON_DEVICE)
        self.assertNotIn("SENTINEL", json.dumps(event.payload))


class TestDevicePathProbeFunction(_ImpersonationTestCase):
    """
    device_path_probe() on its own. verify() calls it for device_reported
    providers only, so it returns provider_matches_on_device there; the other
    reasons and the ranking are this function's contract for any provider.
    """

    def setUp(self):
        super().setUp()
        self._face("alice", VECTORS[b"alice"])
        self._face("bob", VECTORS[b"bob"])
        self.provider = ProviderRegistry.get_provider("face")
        self.device_face = Extracted(vector=list(VECTORS[b"bob-probe"]))

    def test_device_path_disabled(self):
        self._enable()
        probe, reason = device_path_probe(SUBJECT_MODEL, "alice", "face", self.provider, self.device_face)
        self.assertEqual((probe, reason), (None, SKIP_DEVICE_PATH_DISABLED))

    def test_no_device_template(self):
        self._enable(device_path=True)
        for device_template in (None, Extracted(template=b"x")):
            with self.subTest(device_template=device_template):
                probe, reason = device_path_probe(SUBJECT_MODEL, "alice", "face", self.provider, device_template)
                self.assertEqual((probe, reason), (None, SKIP_NO_DEVICE_TEMPLATE))

    def test_opt_in_ranks_the_device_vector(self):
        self._enable(device_path=True)
        probe, reason = device_path_probe(SUBJECT_MODEL, "alice", "face", self.provider, self.device_face)
        self.assertEqual(reason, "")
        self.assertTrue(probe.suspected)
        self.assertEqual(probe.best_match["subject_id"], "bob")


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


class TestDevicePathOverGraphQL(SyntheticSubjectsMixin, _ImpersonationTestCase):

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

        self._enable(modalities=["voice_device"], device_path=True)
        schema = graphene.Schema(query=Query, mutation=Mutation)
        refused = schema.execute(
            'mutation { verifyBiometric(subjectId: "alice", modality: "face", deviceScore: 0.9, '
            "deviceVector: [0.05, 1.0, 0.0]) { verified } }",
            context_value=SimpleNamespace(user=_user(), headers={}),
        )
        self.assertIsInstance(refused.errors[0].original_error, DevicePathRefusedError)

        query = (
            'mutation { verifyBiometric(subjectId: "alice", modality: "voice_device", deviceScore: 60, '
            'deviceTemplate: "dm9pY2U=") { verified impersonationSkipReason impersonation { status } } }'
        )
        result = schema.execute(query, context_value=SimpleNamespace(user=_user(), headers={}))

        self.assertIsNone(result.errors, result.errors)
        data = result.data["verifyBiometric"]
        self.assertEqual(data, {
            "verified": True, "impersonationSkipReason": SKIP_PROVIDER_MATCHES_ON_DEVICE, "impersonation": None,
        })

        BiometricConfig.impersonation_probe = dict(PROBE_DEFAULTS)
        schema.execute(query, context_value=SimpleNamespace(user=_user(), headers={}))
        rows = Query.resolve_biometric_verifications(None, _info(_user()), subject_id="alice")
        self.assertEqual([r.impersonation_skip_reason for r in rows], ["", SKIP_PROVIDER_MATCHES_ON_DEVICE])

    def test_multimodal_leg_device_vector_over_graphql(self):
        import graphene

        from biometric.schema import Mutation

        self._enable(device_path=True)
        query = (
            'mutation { verifyBiometricMultimodal(subjectId: "alice", legs: [{modality: "face", deviceScore: 0.9, '
            "deviceVector: [0.05, 1.0, 0.0]}]) { outcome } }"
        )

        result = graphene.Schema(query=Query, mutation=Mutation).execute(
            query, context_value=SimpleNamespace(user=_user(), headers={}),
        )

        self.assertIsNone((result.data or {}).get("verifyBiometricMultimodal"))
        self.assertIsInstance(result.errors[0].original_error, DevicePathRefusedError)
        self.assertEqual(BiometricVerification.objects.count(), 0)

"""
The device path (deviceScore) is for modalities whose provider matches on the
device (DeviceReportedMatcher). A server-matched modality refuses a device
score before anything is recorded, so a verify caller cannot declare its own
verified=true for it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import graphene

from biometric.models import BiometricMultimodalDecision, BiometricVerification
from biometric.schema import Mutation, Query
from biometric.services import DevicePathRefusedError, verify, verify_multimodal
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase


class TestDevicePathRefusal(_MultimodalServiceTestCase):

    def test_server_matched_modalities_refuse_a_device_score(self):
        for modality in ("face", "fingerprint"):
            with self.subTest(modality):
                with self.assertRaises(DevicePathRefusedError) as raised:
                    verify(SUBJECT_MODEL, "s1", modality, device_score=99.0, actor="agent")
                self.assertEqual(raised.exception.extensions, {"code": "BIOMETRIC_DEVICE_PATH_REFUSED"})
        self.assertEqual(BiometricVerification.objects.count(), 0)

    def test_device_reported_modality_takes_the_device_score(self):
        result = verify(SUBJECT_MODEL, "s1", "voice_device", device_score=60.0, actor="agent")

        self.assertTrue(result.verified)
        self.assertEqual(result.origin, "device")

    def test_multimodal_refuses_before_any_leg_runs(self):
        legs = [
            {"modality": "voice_device", "device_score": 60.0},
            {"modality": "face", "device_score": 0.99},
        ]
        with self.assertRaises(DevicePathRefusedError):
            verify_multimodal(SUBJECT_MODEL, "s1", legs, actor="agent")
        self.assertEqual(BiometricVerification.objects.count(), 0)
        self.assertEqual(BiometricMultimodalDecision.objects.count(), 0)

    def test_graphql_error_code(self):
        user = MagicMock(is_anonymous=False, username="agent")
        user.has_perms.return_value = True
        result = graphene.Schema(query=Query, mutation=Mutation).execute(
            'mutation { verifyBiometric(subjectId: "s1", modality: "face", deviceScore: 0.99) { verified } }',
            context_value=SimpleNamespace(user=user, headers={}),
        )

        self.assertIsNone((result.data or {}).get("verifyBiometric"))
        self.assertIsInstance(result.errors[0].original_error, DevicePathRefusedError)
        self.assertEqual(result.errors[0].extensions, {"code": "BIOMETRIC_DEVICE_PATH_REFUSED"})
        self.assertEqual(BiometricVerification.objects.count(), 0)

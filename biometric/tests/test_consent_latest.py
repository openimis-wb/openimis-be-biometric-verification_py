"""With REQUIRE_CONSENT, the latest consent row for (subject, modality) decides whether enrol() runs."""

from datetime import timedelta

from django.utils import timezone

from biometric.apps import BiometricConfig
from biometric.models import BiometricConsent, BiometricTemplate
from biometric.services import ConsentRequiredError, enrol
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase


class TestLatestConsentWins(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.require_consent = True

    def _consent(self, granted, minutes_ago, modality="face"):
        row = BiometricConsent.objects.create(
            subject_model=SUBJECT_MODEL, subject_id="s1", modality=modality, granted=granted, recorded_by="agent",
        )
        BiometricConsent.objects.filter(id=row.id).update(recorded_at=timezone.now() - timedelta(minutes=minutes_ago))

    def test_a_later_refusal_revokes(self):
        self._consent(True, minutes_ago=10)
        self._consent(False, minutes_ago=1)

        with self.assertRaises(ConsentRequiredError):
            enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent")
        self.assertEqual(BiometricTemplate.objects.count(), 0)

    def test_a_later_grant_restores(self):
        self._consent(False, minutes_ago=10)
        self._consent(True, minutes_ago=1)

        enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent")
        self.assertEqual(BiometricTemplate.objects.count(), 1)

    def test_another_modality_does_not_count(self):
        self._consent(True, minutes_ago=10)
        self._consent(False, minutes_ago=1, modality="fingerprint")

        enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent")
        self.assertEqual(BiometricTemplate.objects.count(), 1)

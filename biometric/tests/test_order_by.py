"""
The connections of the biometric app sort only on an explicit list of the
columns each node exposes. Hidden columns (the impersonation_* match, the
caller's context, payload, detail, the dedupe key) are never sortable, so
orderBy is no oracle on them.
"""

from django.test import TestCase

from biometric.apps import BiometricConfig
from biometric.models import BiometricVerification
from biometric.schema import OrderByRefusedError
from biometric.tests.test_admin_schema import _User
from biometric.tests.test_verification_records import _execute
from biometric.tests.synthetic_subjects import SyntheticSubjectsMixin

ALL_PERMS = ["174003", "174004", "174005"]

CONNECTIONS = {
    "biometricVerificationRecords": (
        ["-createdAt", "subjectId", "score", "verified"],
        ["impersonationSubjectId", "-impersonation_subject_id", "impersonationScore", "context__site",
         "-context", "impersonationEvidence"],
    ),
    "biometricMultimodalDecisions": (
        ["-createdAt", "outcome", "score"],
        ["verificationIds", "reasons", "modalities"],
    ),
    "biometricAuditEvents": (
        ["-sequence", "action", "createdAt"],
        ["payload", "payload__matched_subject_id", "-payload__exclude_subject", "prevHash", "hash"],
    ),
    "biometricAlerts": (
        ["-triggeredAt", "severity", "state"],
        ["detail", "detail__matched_subject_id", "dedupeKey", "triggerEvent__payload"],
    ),
    "biometricErasures": (
        ["-erasedAt", "reason", "subjectId"],
        ["erased", "erased__face"],
    ),
}


class TestOrderByAllowList(SyntheticSubjectsMixin, TestCase):

    def setUp(self):
        super().setUp()
        for name in ("gql_biometric_read_perms", "gql_biometric_audit_perms", "gql_biometric_identify_perms"):
            self.addCleanup(setattr, BiometricConfig, name, getattr(BiometricConfig, name))
        BiometricConfig.gql_biometric_read_perms = ["174004"]
        BiometricConfig.gql_biometric_audit_perms = ["174005"]
        BiometricConfig.gql_biometric_identify_perms = ["174003"]
        BiometricVerification.objects.create(
            subject_model="individual.Individual", subject_id="s1", modality="face", score=0.9, threshold=0.5,
            verified=True, origin="server", actor="agent", impersonation_subject_id="s2",
        )

    def _order(self, field, order):
        query = 'query { %s(first: 5, orderBy: ["%s"]) { totalCount } }' % (field, order)
        return _execute(query, _User(perms=ALL_PERMS))

    def test_listed_columns_sort(self):
        for field, (allowed, _) in CONNECTIONS.items():
            for order in allowed:
                with self.subTest(field=field, order=order):
                    result = self._order(field, order)
                    self.assertIsNone(result.errors, result.errors)

    def test_hidden_columns_are_refused(self):
        for field, (_, refused) in CONNECTIONS.items():
            for order in refused:
                with self.subTest(field=field, order=order):
                    result = self._order(field, order)
                    self.assertIsNone((result.data or {}).get(field))
                    self.assertIsInstance(getattr(result.errors[0], "original_error", None), OrderByRefusedError)
                    self.assertEqual(result.errors[0].extensions, {"code": "BIOMETRIC_ORDER_BY_REFUSED"})

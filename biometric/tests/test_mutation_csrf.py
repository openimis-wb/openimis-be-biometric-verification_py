"""
Every mutation of the biometric app runs core's CSRF check first, as
core.schema.OpenIMISMutation does: outside dev mode and tests, a request
whose X-CSRFToken header differs from the session token is refused before
anything else happens.
"""

from types import SimpleNamespace
from unittest.mock import patch

import graphene
from django.core.exceptions import PermissionDenied
from django.test import override_settings

from biometric.audit_chain import ACTION_VERIFY, record_event
from biometric.models import BiometricAlert, BiometricConsent
from biometric.schema import Mutation, Query
from biometric.tests.test_admin_schema import _User
from biometric.tests.test_audit_schema import SUBJECT_MODEL, _AuditSchemaTestCase

ALL_PERMS = ["174001", "174002", "174005", "174006", "174008"]


def _request(user, header_token):
    return SimpleNamespace(
        user=user, headers={"User-Agent": "browser"}, session={"csrftoken": "session-token"},
        META={"HTTP_X_CSRFTOKEN": header_token},
    )


@override_settings(MODE="prod", IS_TESTING=False, USER_AGENT_CSRF_BYPASS=[])
class TestMutationCsrf(_AuditSchemaTestCase):

    def setUp(self):
        super().setUp()
        from biometric.apps import BiometricConfig

        for name in ("gql_biometric_enrol_perms", "gql_biometric_verify_perms", "gql_biometric_audit_verify_perms"):
            self.addCleanup(setattr, BiometricConfig, name, getattr(BiometricConfig, name))
        BiometricConfig.gql_biometric_enrol_perms = ["174001"]
        BiometricConfig.gql_biometric_verify_perms = ["174002"]
        BiometricConfig.gql_biometric_audit_verify_perms = ["174008"]
        event = record_event(ACTION_VERIFY, actor="agent", subject_model=SUBJECT_MODEL, subject_id="s1",
                             payload={"verified": False})
        self.alert = BiometricAlert.objects.create(
            rule_kind="FAILED_VERIFICATIONS", severity="MEDIUM", title="t", dedupe_key="k",
            subject_model=SUBJECT_MODEL, subject_id="s1", trigger_event=event,
        )
        self.mutations = {
            "enrolBiometric": 'mutation { enrolBiometric(subjectId: "s1", modality: "face", sample: "eA==") { id } }',
            "verifyBiometric": 'mutation { verifyBiometric(subjectId: "s1", modality: "face", sample: "eA==") '
                               "{ verified } }",
            "verifyBiometricMultimodal": 'mutation { verifyBiometricMultimodal(subjectId: "s1", '
                                         'legs: [{modality: "face", sample: "eA=="}]) { outcome } }',
            "recordBiometricConsent": 'mutation { recordBiometricConsent(subjectId: "s1", modality: "face", '
                                      "granted: true) { ok } }",
            "acknowledgeBiometricAlert": 'mutation { acknowledgeBiometricAlert(id: "%s") { state } }' % self.alert.id,
            "resolveBiometricAlert": 'mutation { resolveBiometricAlert(id: "%s") { state } }' % self.alert.id,
            "verifyBiometricAuditChain": "mutation { verifyBiometricAuditChain { ok } }",
        }

    def _execute(self, query, header_token):
        user = _User(perms=ALL_PERMS, username="supervisor")
        return graphene.Schema(query=Query, mutation=Mutation).execute(
            query, context_value=_request(user, header_token),
        )

    def test_a_wrong_token_is_refused_before_the_mutation_runs(self):
        with patch("biometric.services.enrol") as enrol, patch("biometric.services.verify") as verify, \
                patch("biometric.services.verify_multimodal") as verify_multimodal:
            for field, query in self.mutations.items():
                with self.subTest(field):
                    result = self._execute(query, "forged")
                    self.assertIsNone((result.data or {}).get(field))
                    error = getattr(result.errors[0], "original_error", None)
                    self.assertIsInstance(error, PermissionDenied)
                    self.assertEqual(str(error), "CSRF token missing or incorrect.")
        for service in (enrol, verify, verify_multimodal):
            service.assert_not_called()
        self.assertEqual(BiometricConsent.objects.count(), 0)
        self.alert.refresh_from_db()
        self.assertEqual(self.alert.state, "NEW")

    def test_the_session_token_passes(self):
        result = self._execute(self.mutations["acknowledgeBiometricAlert"], "session-token")

        self.assertIsNone(result.errors, result.errors)
        self.assertEqual(result.data["acknowledgeBiometricAlert"], {"state": "ACKNOWLEDGED"})

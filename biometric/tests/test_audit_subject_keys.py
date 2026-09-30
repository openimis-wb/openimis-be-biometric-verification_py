"""
Audit payload keys naming a subject other than the event's own are withheld
from callers without gql_biometric_identify_perms (174003): identify's
exclude_subject and template.consolidate's retired_id.
"""

import json

from biometric.audit_chain import ACTION_CONSOLIDATE, ACTION_IDENTIFY, record_event
from biometric.tests.test_admin_schema import _User
from biometric.tests.test_audit_schema import AUDIT_PERMS, IDENTIFY_PERMS, SUBJECT_MODEL, _AuditSchemaTestCase, _execute

QUERY = "query { biometricAuditEvents(first: 10) { edges { node { action payload } } } }"


class TestSubjectNamingKeys(_AuditSchemaTestCase):

    def setUp(self):
        super().setUp()
        from biometric.apps import BiometricConfig

        self.addCleanup(setattr, BiometricConfig, "gql_biometric_identify_perms",
                        BiometricConfig.gql_biometric_identify_perms)
        BiometricConfig.gql_biometric_identify_perms = list(IDENTIFY_PERMS)
        record_event(ACTION_IDENTIFY, actor="investigator", modality="face", payload={
            "top_k": 5, "scope": None, "exclude_subject": "claimed-subject", "probe": "sample", "matches": [],
        })
        record_event(ACTION_CONSOLIDATE, actor="merger", subject_model=SUBJECT_MODEL, subject_id="kept-subject",
                     payload={"retired_id": "retired-subject", "counts": {"face": 1}, "template_ids": ["t1"]})

    def _payloads(self, perms):
        result = _execute(QUERY, _User(perms=perms))
        self.assertIsNone(result.errors, result.errors)
        return {e["node"]["action"]: json.loads(e["node"]["payload"]) for e in result.data["biometricAuditEvents"]["edges"]}

    def test_withheld_without_the_identify_right(self):
        payloads = self._payloads(AUDIT_PERMS)
        self.assertNotIn("exclude_subject", payloads[ACTION_IDENTIFY])
        self.assertNotIn("retired_id", payloads[ACTION_CONSOLIDATE])
        self.assertEqual(payloads[ACTION_IDENTIFY]["top_k"], 5)
        self.assertEqual(payloads[ACTION_CONSOLIDATE]["counts"], {"face": 1})

    def test_shown_with_the_identify_right(self):
        payloads = self._payloads(AUDIT_PERMS + IDENTIFY_PERMS)
        self.assertEqual(payloads[ACTION_IDENTIFY]["exclude_subject"], "claimed-subject")
        self.assertEqual(payloads[ACTION_CONSOLIDATE]["retired_id"], "retired-subject")

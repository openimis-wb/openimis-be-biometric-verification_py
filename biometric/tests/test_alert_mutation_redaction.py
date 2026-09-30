"""
acknowledgeBiometricAlert / resolveBiometricAlert return the alert's detail
and subject only to a caller holding the audit right (174005); a caller with
the alert right (174006) alone gets them null.
"""

from biometric.audit_chain import ACTION_VERIFY, record_event
from biometric.models import BiometricAlert
from biometric.tests.test_admin_schema import _User
from biometric.tests.test_audit_schema import ALERT_PERMS, AUDIT_PERMS, SUBJECT_MODEL, _AuditSchemaTestCase, _execute

FIELDS = "state detail subjectModel subjectId"
ACK = 'mutation { acknowledgeBiometricAlert(id: "%s") { ' + FIELDS + " } }"
RESOLVE = 'mutation { resolveBiometricAlert(id: "%s") { ' + FIELDS + " } }"


class TestAlertMutationRedaction(_AuditSchemaTestCase):

    def setUp(self):
        super().setUp()
        event = record_event(ACTION_VERIFY, actor="agent", subject_model=SUBJECT_MODEL, subject_id="s1",
                             payload={"verified": False})
        self.alerts = [
            BiometricAlert.objects.create(
                rule_kind="FAILED_VERIFICATIONS", severity="MEDIUM", title="t", dedupe_key=f"k{i}",
                subject_model=SUBJECT_MODEL, subject_id="s1", trigger_event=event,
                detail={"attempts": 3, "actors": ["agent"]},
            )
            for i in range(2)
        ]

    def _both(self, user):
        return (
            _execute(ACK % self.alerts[0].id, user).data["acknowledgeBiometricAlert"],
            _execute(RESOLVE % self.alerts[1].id, user).data["resolveBiometricAlert"],
        )

    def test_alert_right_alone_gets_no_detail_or_subject(self):
        for data in self._both(_User(perms=ALERT_PERMS, username="supervisor")):
            self.assertEqual(data, {"state": data["state"], "detail": None, "subjectModel": None, "subjectId": None})

    def test_audit_right_sees_them(self):
        for data in self._both(_User(perms=ALERT_PERMS + AUDIT_PERMS, username="supervisor")):
            self.assertEqual(data["subjectModel"], SUBJECT_MODEL)
            self.assertEqual(data["subjectId"], "s1")
            self.assertIn('"attempts": 3', data["detail"])

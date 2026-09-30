from unittest.mock import patch


class SyntheticSubjectsMixin:
    """
    For tests whose subjects are bare ids ("s1", "alice") with no subject row,
    or whose users are doubles rather than core users: the subject existence
    and location scope checks (biometric/subjects.py) let everything through.
    test_subject_scope covers those checks against real users and Individuals.
    """

    def setUp(self):
        for target, replacement in (
            ("biometric.subjects.check_subject", lambda user, subject_model, subject_id: None),
            ("biometric.subjects.scope_rows", lambda queryset, user: queryset),
        ):
            patcher = patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        super().setUp()

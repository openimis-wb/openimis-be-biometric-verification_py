"""
The enrol (174001), verify (174002) and read (174004) surfaces act on a
subject only within the caller's location scope, the one individual.Individual
applies to its own list, and only on a subject that exists. Superusers and
IMIS admins bypass the scope. A subject model with no location relation is
only checked for existence.
"""

import base64
from types import SimpleNamespace

import graphene
from django.core.cache import cache
from graphql_relay import to_global_id

from biometric.apps import BiometricConfig
from biometric.models import BiometricConsent, BiometricMultimodalDecision, BiometricTemplate, BiometricVerification
from biometric.schema import Mutation, Query
from biometric.subjects import SUBJECT_MODEL_UNKNOWN, SUBJECT_NOT_FOUND, SubjectRefusedError
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase

RIGHTS = (174001, 174002, 174004)
SAMPLE = base64.b64encode(b"photo").decode()


class _Root(Query, graphene.ObjectType):
    node = graphene.relay.Node.Field()


def _role(name, rights):
    from core.models import Role, RoleRight
    from core.utils import TimeUtils

    role = Role.objects.create(name=name, is_blocked=False, is_system=0, audit_user_id=-1)
    for right_id in rights:
        RoleRight.objects.create(role_id=role.id, right_id=right_id, audit_user_id=-1, validity_from=TimeUtils.now())
    return role


def _individual(user, village):
    from individual.models import Individual

    individual = Individual(first_name="Bio", last_name="Scope", dob="1990-01-01", location=village)
    individual.save(username=user.username)
    return str(individual.id)


class TestSubjectScope(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        from core.test_helpers import create_admin_role, create_test_interactive_user
        from location.test_helpers import assign_user_districts, create_test_village

        for name in ("gql_biometric_enrol_perms", "gql_biometric_verify_perms", "gql_biometric_read_perms"):
            self.addCleanup(setattr, BiometricConfig, name, getattr(BiometricConfig, name))
        BiometricConfig.gql_biometric_enrol_perms = ["174001"]
        BiometricConfig.gql_biometric_verify_perms = ["174002"]
        BiometricConfig.gql_biometric_read_perms = ["174004"]
        cache.clear()
        self.addCleanup(cache.clear)

        village_a = create_test_village({"name": "Bio Village A", "code": "BioViA"})
        village_b = create_test_village({"name": "Bio Village B", "code": "BioViB"})
        self.admin = create_test_interactive_user(username="bioScopeAdmin", roles=[create_admin_role().id])
        self.agent = create_test_interactive_user(
            username="bioScopeAgentA", roles=[_role("Biometric agent", RIGHTS).id],
        )
        assign_user_districts(self.agent, [village_a.parent.parent.code])
        cache.clear()
        self.inside = _individual(self.admin, village_a)
        self.outside = _individual(self.admin, village_b)
        for subject_id in (self.inside, self.outside):
            BiometricTemplate.objects.create(
                subject_model=SUBJECT_MODEL, subject_id=subject_id, modality="fingerprint", kind="template",
                template=b"print", provider="fake_matcher", model_name="",
            )
            BiometricVerification.objects.create(
                subject_model=SUBJECT_MODEL, subject_id=subject_id, modality="fingerprint", score=100.0,
                threshold=50.0, verified=True, origin="server", actor="agent",
            )
            BiometricMultimodalDecision.objects.create(
                subject_model=SUBJECT_MODEL, subject_id=subject_id, outcome="accept", score=1.0, reasons=[],
                modalities=["fingerprint"], verification_ids=[], actor="agent",
            )

    def _execute(self, query, user):
        return graphene.Schema(query=_Root, mutation=Mutation).execute(
            query, context_value=SimpleNamespace(user=user, headers={}),
        )

    def _surfaces(self, subject_id, subject_model=None):
        model = ', subjectModel: "%s"' % subject_model if subject_model else ""
        return {
            "enrolBiometric": 'mutation { enrolBiometric(subjectId: "%s"%s, modality: "fingerprint", sample: "%s") '
                              "{ id } }" % (subject_id, model, SAMPLE),
            "recordBiometricConsent": 'mutation { recordBiometricConsent(subjectId: "%s"%s, modality: "face", '
                                      "granted: true) { ok } }" % (subject_id, model),
            "verifyBiometric": 'mutation { verifyBiometric(subjectId: "%s"%s, modality: "fingerprint", sample: "%s") '
                               "{ verified } }" % (subject_id, model, SAMPLE),
            "verifyBiometricMultimodal": 'mutation { verifyBiometricMultimodal(subjectId: "%s"%s, legs: '
                                         '[{modality: "fingerprint", sample: "%s"}]) { outcome } }'
                                         % (subject_id, model, SAMPLE),
            "biometricTemplates": 'query { biometricTemplates(subjectId: "%s"%s) { id } }' % (subject_id, model),
            "biometricVerifications": 'query { biometricVerifications(subjectId: "%s"%s) { id } }'
                                      % (subject_id, model),
        }

    def _assert_refused(self, result, field, code):
        self.assertIsNone((result.data or {}).get(field))
        error = getattr(result.errors[0], "original_error", None)
        self.assertIsInstance(error, SubjectRefusedError)
        self.assertEqual(result.errors[0].extensions, {"code": code})

    def _counts(self):
        return (BiometricTemplate.objects.count(), BiometricVerification.objects.count(),
                BiometricConsent.objects.count(), BiometricMultimodalDecision.objects.count())

    def test_a_subject_outside_the_district_is_refused(self):
        before = self._counts()
        for field, query in self._surfaces(self.outside).items():
            with self.subTest(field):
                self._assert_refused(self._execute(query, self.agent), field, SUBJECT_NOT_FOUND)
        self.assertEqual(self._counts(), before)

    def test_a_subject_inside_the_district_is_served(self):
        for field, query in self._surfaces(self.inside).items():
            with self.subTest(field):
                result = self._execute(query, self.agent)
                self.assertIsNone(result.errors, result.errors)

    def test_an_admin_reaches_every_subject(self):
        for field, query in self._surfaces(self.outside).items():
            with self.subTest(field):
                self.assertIsNone(self._execute(query, self.admin).errors)

    def test_a_missing_subject_or_model_is_refused(self):
        cases = {
            SUBJECT_NOT_FOUND: ("00000000-0000-0000-0000-000000000000", None),
            SUBJECT_MODEL_UNKNOWN: (self.inside, "nothing.Here"),
        }
        for code, (subject_id, model) in cases.items():
            for field, query in self._surfaces(subject_id, model).items():
                with self.subTest(code=code, field=field):
                    self._assert_refused(self._execute(query, self.admin), field, code)
        self._assert_refused(
            self._execute(self._surfaces("not-a-uuid")["biometricTemplates"], self.admin),
            "biometricTemplates", SUBJECT_NOT_FOUND,
        )

    def test_a_model_without_location_is_only_checked_for_existence(self):
        role = _role("Biometric unscoped subject", ())
        for field, query in self._surfaces(str(role.id), "core.Role").items():
            with self.subTest(field):
                result = self._execute(query, self.agent)
                self.assertIsNone(result.errors, result.errors)

    def test_connections_and_node_lookups_hide_out_of_scope_rows(self):
        connections = {
            "biometricVerificationRecords": ("BiometricVerificationGQLType", BiometricVerification),
            "biometricMultimodalDecisions": ("BiometricMultimodalDecisionGQLType", BiometricMultimodalDecision),
        }
        for field, (type_name, model) in connections.items():
            with self.subTest(field):
                result = self._execute("query { %s(first: 10) { edges { node { subjectId } } } }" % field, self.agent)
                self.assertIsNone(result.errors, result.errors)
                self.assertEqual([e["node"]["subjectId"] for e in result.data[field]["edges"]], [self.inside])

                admin = self._execute("query { %s(first: 10) { totalCount } }" % field, self.admin)
                self.assertEqual(admin.data[field]["totalCount"], 2)

                for subject_id, visible in ((self.inside, True), (self.outside, False)):
                    row = model.objects.get(subject_id=subject_id)
                    lookup = self._execute(
                        'query { node(id: "%s") { ... on %s { subjectId } } }'
                        % (to_global_id(type_name, row.id), type_name),
                        self.agent,
                    )
                    self.assertEqual(bool((lookup.data or {}).get("node")), visible)

    def test_rows_of_unscoped_and_unknown_models(self):
        role = _role("Biometric unscoped row", ())
        for subject_model, subject_id in (("core.Role", str(role.id)), ("nothing.Here", "x1")):
            BiometricVerification.objects.create(
                subject_model=subject_model, subject_id=subject_id, modality="fingerprint", score=1.0,
                threshold=50.0, verified=False, origin="server", actor="agent",
            )
        query = "query { biometricVerificationRecords(first: 10) { edges { node { subjectModel subjectId } } } }"

        def seen(user):
            result = self._execute(query, user)
            self.assertIsNone(result.errors, result.errors)
            return sorted((e["node"]["subjectModel"], e["node"]["subjectId"])
                          for e in result.data["biometricVerificationRecords"]["edges"])

        self.assertEqual(seen(self.agent), sorted([("core.Role", str(role.id)), (SUBJECT_MODEL, self.inside)]))
        self.assertIn(("nothing.Here", "x1"), seen(self.admin))

    def test_a_refused_template_list_records_no_event(self):
        from biometric.audit_chain import ACTION_TEMPLATE_LIST
        from biometric.models import BiometricAuditEvent
        from biometric.tests.test_audit_chain import restore_audit_settings_on_cleanup

        restore_audit_settings_on_cleanup(self)
        BiometricConfig.audit = {"enabled": True, "rules": {}}

        self._execute(self._surfaces(self.outside)["biometricTemplates"], self.agent)
        self.assertFalse(BiometricAuditEvent.objects.filter(action=ACTION_TEMPLATE_LIST).exists())
        self._execute(self._surfaces(self.inside)["biometricTemplates"], self.agent)
        self.assertTrue(BiometricAuditEvent.objects.filter(action=ACTION_TEMPLATE_LIST).exists())

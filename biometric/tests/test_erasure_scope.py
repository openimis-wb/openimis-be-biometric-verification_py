"""
biometricErasures, its node lookup and biometricErasureFilterValues show a
non-admin only the tombstones whose subject still exists, is not soft-deleted
and lies in the caller's location scope, plus the tombstones naming no subject
(docs/wb-biometric-dedup-seam.md §6.12, §6.15). Admins see every tombstone.
"""

import datetime
import uuid
from types import SimpleNamespace

import graphene
from django.core.cache import cache
from django.utils import timezone
from graphql_relay import to_global_id

from biometric import services
from biometric.apps import BiometricConfig
from biometric.models import BiometricErasure, BiometricRetentionPolicy
from biometric.schema import Mutation, Query
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase
from biometric.tests.test_subject_scope import _individual, _role

AUDIT = 174005
ERASURE_TYPE = "BiometricErasureGQLType"
UNSCOPED_MODEL = "core.User"


class _Root(Query, graphene.ObjectType):
    node = graphene.relay.Node.Field()


class TestErasureScope(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        from core.test_helpers import create_admin_role, create_test_interactive_user
        from individual.models import Individual
        from location.test_helpers import assign_user_districts, create_test_village

        self.addCleanup(setattr, BiometricConfig, "gql_biometric_audit_perms", BiometricConfig.gql_biometric_audit_perms)
        BiometricConfig.gql_biometric_audit_perms = [str(AUDIT)]
        cache.clear()
        self.addCleanup(cache.clear)

        village_a = create_test_village({"name": "Erasure Village A", "code": "EraViA"})
        village_b = create_test_village({"name": "Erasure Village B", "code": "EraViB"})
        self.admin = create_test_interactive_user(username="erasureScopeAdmin", roles=[create_admin_role().id])
        self.agent = create_test_interactive_user(
            username="erasureScopeAgentA", roles=[_role("Erasure auditor", (AUDIT,)).id],
        )
        assign_user_districts(self.agent, [village_a.parent.parent.code])
        cache.clear()

        inside = _individual(self.admin, village_a)
        deleted = _individual(self.admin, village_a)
        outside = _individual(self.admin, village_b)
        soft_deleted = Individual.objects.get(id=deleted)
        soft_deleted.is_deleted = True
        soft_deleted.save(username=self.admin.username)

        # The in-scope tombstone comes from the real erase path, so its subject_id is the text purge() stores.
        services.enrol(SUBJECT_MODEL, inside, "face", b"photo-bytes", actor="tester")
        BiometricRetentionPolicy.objects.create(
            template_retention_days=None, purge_enabled=False,
            active_template_retention_days=0, purge_active_enabled=True,
        )
        tombstone = services.purge(now=timezone.now() + datetime.timedelta(days=1), actor="inside")
        self.assertEqual(tombstone.subject_id, inside)

        for erased_by, subject_model, subject_id in (
            ("outside", SUBJECT_MODEL, outside),
            ("deleted", SUBJECT_MODEL, deleted),
            ("missing", SUBJECT_MODEL, str(uuid.uuid4())),
            ("none", "", ""),
            ("unscopedExisting", UNSCOPED_MODEL, str(self.admin.id)),
            ("unscopedMissing", UNSCOPED_MODEL, str(uuid.uuid4())),
        ):
            BiometricErasure.objects.create(
                subject_model=subject_model, subject_id=subject_id, modalities=["face"], erased={"face": 1},
                reason="request", erased_by=erased_by,
            )
        self.erasures = {row.erased_by: row for row in BiometricErasure.objects.all()}

    def _execute(self, query, user):
        return graphene.Schema(query=_Root, mutation=Mutation).execute(
            query, context_value=SimpleNamespace(user=user, headers={}),
        )

    def _listed(self, user, args="first: 20"):
        result = self._execute("query { biometricErasures(%s) { totalCount edges { node { erasedBy } } } }" % args, user)
        self.assertIsNone(result.errors, result.errors)
        data = result.data["biometricErasures"]
        return data["totalCount"], sorted(edge["node"]["erasedBy"] for edge in data["edges"])

    def test_a_district_auditor_sees_live_subjects_of_the_district_and_rows_without_subject(self):
        self.assertEqual(self._listed(self.agent), (3, ["inside", "none", "unscopedExisting"]))

    def test_an_admin_sees_every_tombstone(self):
        self.assertEqual(self._listed(self.admin), (len(self.erasures), sorted(self.erasures)))

    def test_filtering_on_an_out_of_scope_subject_returns_nothing(self):
        args = 'subjectId: "%s", first: 5' % self.erasures["outside"].subject_id
        self.assertEqual(self._listed(self.agent, args), (0, []))
        self.assertEqual(self._listed(self.admin, args), (1, ["outside"]))

    def test_node_lookup_hides_out_of_scope_tombstones(self):
        for name, row in self.erasures.items():
            visible = name in ("inside", "none", "unscopedExisting")
            with self.subTest(name):
                query = 'query { node(id: "%s") { ... on %s { erasedBy } } }' % (
                    to_global_id(ERASURE_TYPE, row.id), ERASURE_TYPE,
                )
                self.assertEqual(bool((self._execute(query, self.agent).data or {}).get("node")), visible)
                self.assertTrue((self._execute(query, self.admin).data or {}).get("node"))

    def test_filter_values_come_from_the_visible_tombstones(self):
        query = "query { biometricErasureFilterValues { erasedBy subjectModel } }"
        agent = self._execute(query, self.agent)
        self.assertIsNone(agent.errors, agent.errors)
        self.assertEqual(agent.data["biometricErasureFilterValues"], {
            "erasedBy": ["inside", "none", "unscopedExisting"],
            "subjectModel": ["", UNSCOPED_MODEL, SUBJECT_MODEL],
        })
        admin = self._execute(query, self.admin)
        self.assertIsNone(admin.errors, admin.errors)
        self.assertEqual(admin.data["biometricErasureFilterValues"]["erasedBy"], sorted(self.erasures))

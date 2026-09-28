"""
Admin GraphQL surface (docs/wb-biometric-dedup-seam.md §6.12):
biometricDecisionCriteria, biometricRetentionPolicy, biometricErasures,
biometricAuditChainStatus and verifyBiometricAuditChain. Rights, whitelisted
fields, pagination and the stored chain check.
"""

import datetime
import json
from types import SimpleNamespace

import graphene
from cryptography.fernet import Fernet
from django.core.exceptions import PermissionDenied
from django.db import connection
from django.test import SimpleTestCase, TestCase

from biometric import services
from biometric.apps import DEFAULT_CFG, BiometricConfig, _SETTINGS_KEY_MAP
from biometric.audit_chain import ACTION_VERIFY, latest_chain_check, record_chain_check, record_event
from biometric.models import (
    BiometricAuditChainCheck,
    BiometricAuditEvent,
    BiometricErasure,
    BiometricRetentionPolicy,
)
from biometric.risk_profiles import decision_criteria
from biometric.schema import Mutation, Query
from biometric.tests.test_audit_chain import AuditConfigMixin

CONFIG_PERMS = ["174007"]
AUDIT_PERMS = ["174005"]
AUDIT_VERIFY_PERMS = ["174008"]

ADMIN_SETTINGS = (
    "fusion", "modalities", "risk_profiles", "template_key",
    "gql_biometric_config_perms", "gql_biometric_audit_perms", "gql_biometric_audit_verify_perms",
)


class _User:

    def __init__(self, perms=(), username="admin", anonymous=False):
        self.perms = set(perms)
        self.username = username
        self.is_anonymous = anonymous
        self.is_authenticated = not anonymous
        self.id = None if anonymous else 1

    def has_perms(self, perms):
        return all(p in self.perms for p in perms)


def _execute(query, user):
    schema = graphene.Schema(query=Query, mutation=Mutation)
    return schema.execute(query, context_value=SimpleNamespace(user=user, headers={}))


def _denied(result):
    return bool(result.errors) and isinstance(getattr(result.errors[0], "original_error", None), PermissionDenied)


class _AdminConfigMixin:

    def setUp(self):
        for name in ADMIN_SETTINGS:
            self.addCleanup(setattr, BiometricConfig, name, getattr(BiometricConfig, name))
        super().setUp()
        BiometricConfig.gql_biometric_config_perms = list(CONFIG_PERMS)
        BiometricConfig.gql_biometric_audit_perms = list(AUDIT_PERMS)
        BiometricConfig.gql_biometric_audit_verify_perms = list(AUDIT_VERIFY_PERMS)
        BiometricConfig.modalities = {
            "face": {"provider": "fake_embedding", "threshold": 0.5, "provider_config": {"api_key": "SENTINEL-KEY"}},
            "fingerprint": {"provider": "fake_matcher", "threshold": 50.0},
        }
        BiometricConfig.fusion = {
            "weights": {"face": 1.0, "fingerprint": 2.0},
            "thresholds": {"accept": 0.7, "review": 0.6},
            "floors": {"fingerprint": 20.0},
            "floor_decision": "review",
            "endpoint_token": "SENTINEL-TOKEN",
        }
        BiometricConfig.risk_profiles = {}


class TestRights(SimpleTestCase):

    def test_defaults_and_settings_keys(self):
        self.assertEqual(DEFAULT_CFG["gql_biometric_config_perms"], ["174007"])
        self.assertEqual(DEFAULT_CFG["gql_biometric_audit_verify_perms"], ["174008"])
        self.assertEqual(BiometricConfig.gql_biometric_config_perms, ["174007"])
        self.assertEqual(BiometricConfig.gql_biometric_audit_verify_perms, ["174008"])
        self.assertEqual(_SETTINGS_KEY_MAP["GQL_BIOMETRIC_CONFIG_PERMS"], "gql_biometric_config_perms")
        self.assertEqual(_SETTINGS_KEY_MAP["GQL_BIOMETRIC_AUDIT_VERIFY_PERMS"], "gql_biometric_audit_verify_perms")


CRITERIA_QUERY = """
query {
  biometricDecisionCriteria {
    base {
      acceptThreshold reviewThreshold floorDecision required
      floors { modality value } modalityThresholds { modality value } weights { modality value }
    }
    profiles {
      name valid errors
      overrides {
        acceptThreshold reviewThreshold floorDecision required
        floors { modality value } modalityThresholds { modality value }
      }
      effective {
        acceptThreshold reviewThreshold floorDecision required
        floors { modality value } modalityThresholds { modality value } weights { modality value }
      }
    }
  }
}
"""

RETENTION_QUERY = """
query { biometricRetentionPolicy { templateRetentionDays purgeEnabled activeTemplateRetentionDays purgeActiveEnabled } }
"""

ERASURES_QUERY = """
query {
  biometricErasures(%s) {
    totalCount
    pageInfo { hasNextPage }
    edges { node { id subjectModel subjectId modalities erased reason erasedBy erasedAt } }
  }
}
"""

STATUS_QUERY = """
query {
  biometricAuditChainStatus {
    id ok checkedAt checkedBy checked headSequence headHash divergenceKind divergenceSequence divergenceDetail
  }
}
"""

VERIFY_MUTATION = """
mutation {
  verifyBiometricAuditChain {
    ok checkedBy checked headSequence headHash divergenceKind divergenceSequence
  }
}
"""


def _values(pairs):
    return {pair["modality"]: pair["value"] for pair in pairs}


class TestDecisionCriteria(_AdminConfigMixin, SimpleTestCase):

    def _data(self, user=None):
        result = _execute(CRITERIA_QUERY, user or _User(perms=CONFIG_PERMS))
        self.assertIsNone(result.errors, result.errors)
        return result.data["biometricDecisionCriteria"]

    def test_rights(self):
        self.assertTrue(_denied(_execute(CRITERIA_QUERY, _User(anonymous=True))))
        self.assertTrue(_denied(_execute(CRITERIA_QUERY, _User(perms=["174004", "174005", "174002"]))))

    def test_base_rules(self):
        base = self._data()["base"]

        self.assertEqual((base["acceptThreshold"], base["reviewThreshold"]), (0.7, 0.6))
        self.assertEqual(base["floorDecision"], "review")
        self.assertEqual(base["required"], [])
        self.assertEqual(_values(base["floors"]), {"fingerprint": 20.0})
        self.assertEqual(_values(base["modalityThresholds"]), {"face": 0.5, "fingerprint": 50.0})
        self.assertEqual(_values(base["weights"]), {"face": 1.0, "fingerprint": 2.0})

    def test_valid_profile_with_overrides_and_effective_rules(self):
        BiometricConfig.risk_profiles = {
            "high_risk": {
                "thresholds": {"accept": 0.9},
                "floors": {"face": 0.4},
                "floor_decision": "reject",
                "required": ["fingerprint"],
                "modality_thresholds": {"face": 0.8},
            },
        }

        profile = self._data()["profiles"][0]

        self.assertEqual((profile["name"], profile["valid"], profile["errors"]), ("high_risk", True, []))
        overrides = profile["overrides"]
        self.assertEqual((overrides["acceptThreshold"], overrides["reviewThreshold"]), (0.9, None))
        self.assertEqual(overrides["floorDecision"], "reject")
        self.assertEqual(overrides["required"], ["fingerprint"])
        self.assertEqual(_values(overrides["floors"]), {"face": 0.4})
        self.assertEqual(_values(overrides["modalityThresholds"]), {"face": 0.8})
        effective = profile["effective"]
        self.assertEqual((effective["acceptThreshold"], effective["reviewThreshold"]), (0.9, 0.6))
        self.assertEqual(effective["floorDecision"], "reject")
        self.assertEqual(effective["required"], ["fingerprint"])
        self.assertEqual(_values(effective["floors"]), {"fingerprint": 20.0, "face": 0.4})
        self.assertEqual(_values(effective["modalityThresholds"]), {"face": 0.8, "fingerprint": 50.0})
        self.assertEqual(_values(effective["weights"]), {"face": 1.0, "fingerprint": 2.0})

    def test_invalid_profile_is_listed_with_its_errors_and_no_effective_rules(self):
        BiometricConfig.risk_profiles = {
            "b_loose": {"thresholds": {"accept": 0.1}},
            "c_garbage": {"thresholds": {"accept": "high"}, "floors": {"face": [1]}, "floor_decision": "maybe",
                          "required": "fingerprint", "weights": {"face": 9}, "api_key": "SENTINEL-PROFILE"},
            "a_ok": {"floor_decision": "reject"},
        }

        profiles = self._data()["profiles"]

        self.assertEqual([p["name"] for p in profiles], ["a_ok", "b_loose", "c_garbage"])
        self.assertTrue(profiles[0]["valid"])
        loose, garbage = profiles[1], profiles[2]
        self.assertFalse(loose["valid"])
        self.assertIsNone(loose["effective"])
        self.assertEqual(loose["overrides"]["acceptThreshold"], 0.1)
        self.assertTrue(any("below the base" in e for e in loose["errors"]))
        self.assertFalse(garbage["valid"])
        self.assertIsNone(garbage["effective"])
        self.assertIsNone(garbage["overrides"]["acceptThreshold"])
        self.assertIsNone(garbage["overrides"]["floorDecision"])
        self.assertEqual(garbage["overrides"]["required"], [])
        self.assertEqual(_values(garbage["overrides"]["floors"]), {"face": None})
        self.assertTrue(any("'weights' is not overridable" in e for e in garbage["errors"]))

    def test_no_secret_or_unlisted_config_value_is_exposed(self):
        BiometricConfig.template_key = Fernet.generate_key().decode()
        BiometricConfig.risk_profiles = {"p": {"thresholds": {"accept": 0.9}, "api_key": "SENTINEL-PROFILE"}}

        dumped = json.dumps(self._data())

        for secret in (BiometricConfig.template_key, "SENTINEL-KEY", "SENTINEL-TOKEN", "SENTINEL-PROFILE",
                       "provider_config", "endpoint_token", "fake_embedding"):
            self.assertNotIn(secret, dumped)

    def test_service_function_matches_the_query(self):
        BiometricConfig.risk_profiles = {"p": {"thresholds": {"accept": 0.9}}}
        criteria = decision_criteria()
        self.assertEqual(criteria["base"]["thresholds"], {"accept": 0.7, "review": 0.6})
        self.assertEqual(criteria["profiles"][0]["effective"]["thresholds"], {"accept": 0.9, "review": 0.6})


class TestRetentionPolicy(_AdminConfigMixin, TestCase):

    def test_rights(self):
        self.assertTrue(_denied(_execute(RETENTION_QUERY, _User(anonymous=True))))
        self.assertTrue(_denied(_execute(RETENTION_QUERY, _User(perms=AUDIT_PERMS))))

    def test_null_without_a_policy_row(self):
        result = _execute(RETENTION_QUERY, _User(perms=CONFIG_PERMS))
        self.assertIsNone(result.errors, result.errors)
        self.assertIsNone(result.data["biometricRetentionPolicy"])

    def test_the_row_purge_reads(self):
        BiometricRetentionPolicy.objects.create(
            template_retention_days=30, purge_enabled=True, active_template_retention_days=365,
            purge_active_enabled=False,
        )

        result = _execute(RETENTION_QUERY, _User(perms=CONFIG_PERMS))

        self.assertIsNone(result.errors, result.errors)
        self.assertEqual(result.data["biometricRetentionPolicy"], {
            "templateRetentionDays": 30, "purgeEnabled": True,
            "activeTemplateRetentionDays": 365, "purgeActiveEnabled": False,
        })
        self.assertEqual(services.current_retention_policy().template_retention_days, 30)


class TestErasures(_AdminConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        base = datetime.datetime(2026, 9, 1, 12, 0, 0)
        for index, (subject, reason) in enumerate([("s1", "retention"), ("s2", "ACTIVE_AGE"), ("s3", "retention")]):
            erasure = BiometricErasure.objects.create(
                subject_model="individual.Individual", subject_id=subject, modalities=["face"],
                erased={"face": index + 1}, reason=reason, erased_by="retention",
            )
            BiometricErasure.objects.filter(id=erasure.id).update(erased_at=base + datetime.timedelta(days=index))
        self.auditor = _User(perms=AUDIT_PERMS)

    def _data(self, args):
        result = _execute(ERASURES_QUERY % args, self.auditor)
        self.assertIsNone(result.errors, result.errors)
        return result.data["biometricErasures"]

    def test_rights(self):
        self.assertTrue(_denied(_execute(ERASURES_QUERY % "first: 5", _User(anonymous=True))))
        self.assertTrue(_denied(_execute(ERASURES_QUERY % "first: 5", _User(perms=CONFIG_PERMS + ["174004"]))))

    def test_newest_first_paginated(self):
        data = self._data("first: 2")

        self.assertEqual(data["totalCount"], 3)
        self.assertTrue(data["pageInfo"]["hasNextPage"])
        nodes = [edge["node"] for edge in data["edges"]]
        self.assertEqual([n["subjectId"] for n in nodes], ["s3", "s2"])
        self.assertEqual(nodes[0]["modalities"], ["face"])
        self.assertEqual(json.loads(nodes[0]["erased"]), {"face": 3})
        self.assertEqual((nodes[0]["reason"], nodes[0]["erasedBy"]), ("retention", "retention"))

    def test_filters(self):
        cases = {
            'subjectId: "s2"': ["s2"],
            'reason: "retention"': ["s3", "s1"],
            'erasedAt_Gte: "2026-09-02T00:00:00"': ["s3", "s2"],
            'erasedAt_Lte: "2026-09-02T00:00:00"': ["s1"],
            'erasedBy: "retention", subjectModel: "individual.Individual"': ["s3", "s2", "s1"],
        }
        for args, expected in cases.items():
            with self.subTest(args):
                data = self._data(args + ", first: 10")
                self.assertEqual([e["node"]["subjectId"] for e in data["edges"]], expected)


class TestAuditChainStatus(_AdminConfigMixin, AuditConfigMixin, TestCase):

    def setUp(self):
        super().setUp()
        BiometricConfig.audit = {"enabled": True, "rules": {}}
        for index in range(3):
            record_event(ACTION_VERIFY, actor="agent", subject_model="individual.Individual",
                         subject_id=f"s{index}", modality="face", payload={"verified": True})
        self.admin = _User(perms=AUDIT_PERMS + AUDIT_VERIFY_PERMS, username="chain-admin")

    def test_rights(self):
        self.assertTrue(_denied(_execute(STATUS_QUERY, _User(anonymous=True))))
        self.assertTrue(_denied(_execute(STATUS_QUERY, _User(perms=AUDIT_VERIFY_PERMS + CONFIG_PERMS))))
        for perms in ([], AUDIT_PERMS, AUDIT_VERIFY_PERMS):
            with self.subTest(perms=perms):
                self.assertTrue(_denied(_execute(VERIFY_MUTATION, _User(perms=perms))))
        self.assertEqual(BiometricAuditChainCheck.objects.count(), 0)

    def test_status_is_null_before_any_check(self):
        result = _execute(STATUS_QUERY, _User(perms=AUDIT_PERMS))
        self.assertIsNone(result.errors, result.errors)
        self.assertIsNone(result.data["biometricAuditChainStatus"])

    def test_verify_stores_an_intact_result_and_appends_no_event(self):
        result = _execute(VERIFY_MUTATION, self.admin)

        self.assertIsNone(result.errors, result.errors)
        checked = result.data["verifyBiometricAuditChain"]
        head = BiometricAuditEvent.objects.order_by("-sequence").first()
        self.assertEqual(checked, {
            "ok": True, "checkedBy": "chain-admin", "checked": 3, "headSequence": 3, "headHash": head.hash,
            "divergenceKind": "", "divergenceSequence": None,
        })
        self.assertEqual(BiometricAuditEvent.objects.count(), 3)

        status = _execute(STATUS_QUERY, _User(perms=AUDIT_PERMS)).data["biometricAuditChainStatus"]
        self.assertTrue(status["ok"])
        self.assertEqual((status["headSequence"], status["checked"]), (3, 3))
        self.assertIsNotNone(status["checkedAt"])

    def test_broken_chain_reports_the_first_divergence_and_the_latest_check_wins(self):
        record_chain_check(actor="first")
        with connection.cursor() as cursor:
            cursor.execute("UPDATE biometric_audit_event SET payload = %s::jsonb WHERE sequence = 2",
                           ['{"verified": false}'])

        result = _execute(VERIFY_MUTATION, self.admin)

        self.assertIsNone(result.errors, result.errors)
        status = _execute(STATUS_QUERY, _User(perms=AUDIT_PERMS)).data["biometricAuditChainStatus"]
        self.assertFalse(status["ok"])
        self.assertEqual((status["divergenceKind"], status["divergenceSequence"]), ("altered_row", 2))
        self.assertEqual((status["checked"], status["headSequence"]), (1, 3))
        self.assertEqual(status["checkedBy"], "chain-admin")
        self.assertIn("recomputed hash", status["divergenceDetail"])
        self.assertEqual(latest_chain_check().checked_by, "chain-admin")
        self.assertEqual(BiometricAuditChainCheck.objects.count(), 2)

    def test_empty_chain_is_intact(self):
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM biometric_audit_event")

        check = record_chain_check(actor="admin")

        self.assertEqual((check.ok, check.checked, check.head_sequence), (True, 0, 0))
        self.assertEqual(check.head_hash, "0" * 64)

"""
Multimodal verification (docs/wb-biometric-dedup-seam.md §6.8, §6.11):
services.verify_multimodal() verifies each leg with verify() and fuses the leg
scores with fuse() under one risk profile, so every profile key
(thresholds, floors, floor_decision, required, modality_thresholds) applies on
a path a caller reaches, including verifyBiometricMultimodal over GraphQL.
Device-score legs keep the raw scores exact.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import graphene
from django.core.exceptions import PermissionDenied

from biometric.apps import BiometricConfig
from biometric.models import BiometricVerification
from biometric.risk_profiles import RiskProfileError, UnknownRiskProfileError
from biometric.schema import Mutation, Query, VerifyBiometricMultimodalMutation
from biometric.services import _OUTCOME_RANK, verify_multimodal
from biometric.tests.test_risk_profiles import ACCEPTED_PROFILES, BASE_FUSION, BASE_MODALITIES
from biometric.tests.test_services import SUBJECT_MODEL, _MultimodalServiceTestCase

FUSION = {
    "weights": {"face": 1.0, "fingerprint": 1.0},
    "thresholds": {"accept": 0.7, "review": 0.6},
    "floors": {"fingerprint": 20.0},
    "floor_decision": "review",
}


def _legs(face=None, fingerprint=None):
    legs = []
    if face is not None:
        legs.append({"modality": "face", "device_score": face})
    if fingerprint is not None:
        legs.append({"modality": "fingerprint", "device_score": fingerprint})
    return legs


class _MultimodalTestCase(_MultimodalServiceTestCase):

    def setUp(self):
        super().setUp()
        # Device-score legs need a device_reported provider (verify(), §3.4).
        BiometricConfig.modalities = {
            "face": {"provider": "device_reported", "threshold": 0.5},
            "fingerprint": {"provider": "device_reported", "threshold": 50.0},
        }
        BiometricConfig.fusion = dict(FUSION)
        BiometricConfig.risk_profiles = {}

    def _pair(self, profile, legs):
        """(outcome without a profile, outcome under the profile) for the same legs."""
        BiometricConfig.risk_profiles = {"p": profile}
        base = verify_multimodal(SUBJECT_MODEL, "s1", legs, actor="agent")
        profiled = verify_multimodal(SUBJECT_MODEL, "s1", legs, actor="agent", risk_profile="p")
        return base, profiled


class TestEachProfileKeyTakesEffect(_MultimodalTestCase):

    def test_accept_threshold(self):
        base, profiled = self._pair({"thresholds": {"accept": 1.2}}, _legs(0.5, 50.0))
        self.assertEqual((base.decision.outcome, profiled.decision.outcome), ("accept", "review"))
        self.assertEqual(profiled.decision.risk_profile, "p")
        self.assertEqual(base.decision.risk_profile, "")

    def test_review_threshold(self):
        base, profiled = self._pair({"thresholds": {"review": 0.68}}, _legs(0.325, 32.5))
        self.assertEqual((base.decision.outcome, profiled.decision.outcome), ("review", "reject"))

    def test_floors(self):
        base, profiled = self._pair({"floors": {"face": 0.55}}, _legs(0.5, 50.0))
        self.assertEqual((base.decision.outcome, profiled.decision.outcome), ("accept", "review"))
        self.assertIn("'face' score 0.5 below floor 0.55", profiled.decision.reasons)

    def test_floor_decision(self):
        base, profiled = self._pair({"floor_decision": "reject"}, _legs(0.75, 15.0))
        self.assertEqual((base.decision.outcome, profiled.decision.outcome), ("review", "reject"))

    def test_required(self):
        base, profiled = self._pair({"required": ["fingerprint"]}, _legs(face=0.5))
        self.assertEqual((base.decision.outcome, profiled.decision.outcome), ("accept", "review"))
        self.assertIn("required modality 'fingerprint' has no score", profiled.decision.reasons)

    def test_modality_thresholds_raise_the_leg_threshold_and_the_normalisation(self):
        base, profiled = self._pair({"modality_thresholds": {"face": 1.0}}, _legs(0.55, 40.0))
        self.assertEqual((base.decision.outcome, profiled.decision.outcome), ("accept", "review"))
        base_face, profiled_face = base.legs[0], profiled.legs[0]
        self.assertEqual((base_face.threshold, base_face.verified), (0.5, True))
        self.assertEqual((profiled_face.threshold, profiled_face.verified), (1.0, False))
        self.assertEqual(profiled_face.risk_profile, "p")


class TestLegsAndRecords(_MultimodalTestCase):

    def test_each_leg_is_a_recorded_verification(self):
        result = verify_multimodal(SUBJECT_MODEL, "s1", _legs(0.5, 50.0), actor="agent", device_id="tab-1",
                                   context={"site": "koza"}, fallback=True)

        self.assertEqual([leg.modality for leg in result.legs], ["face", "fingerprint"])
        rows = BiometricVerification.objects.filter(subject_id="s1").order_by("modality")
        self.assertEqual([(r.modality, r.score, r.origin, r.device_id, r.fallback, r.actor) for r in rows], [
            ("face", 0.5, "device", "tab-1", True, "agent"),
            ("fingerprint", 50.0, "device", "tab-1", True, "agent"),
        ])
        self.assertEqual(rows[0].context, {"site": "koza"})
        self.assertAlmostEqual(result.decision.score, 1.0)

    def test_server_path_leg(self):
        from biometric.registry import ProviderRegistry
        from biometric.services import enrol

        BiometricConfig.modalities = {**BiometricConfig.modalities,
                                      "face": {"provider": "fake_embedding", "threshold": 0.5}}
        ProviderRegistry._modality_instances.clear()
        enrol(SUBJECT_MODEL, "s1", "face", b"photo", actor="agent")
        result = verify_multimodal(
            SUBJECT_MODEL, "s1",
            [{"modality": "face", "sample": b"photo"}, {"modality": "fingerprint", "device_score": 50.0}],
            actor="agent",
        )

        self.assertEqual(result.legs[0].origin, "server")
        self.assertAlmostEqual(result.legs[0].confidence, 1.0)
        self.assertEqual(result.decision.outcome, "accept")

    def test_invalid_legs_raise_before_any_row(self):
        cases = {
            "no legs": [],
            "not a list": "face",
            "no modality": [{"device_score": 1.0}],
            "duplicate modality": _legs(0.5) + _legs(0.6),
            "neither sample nor score": [{"modality": "face"}],
            "both sample and score": [{"modality": "face", "sample": b"x", "device_score": 1.0}],
            "device template with a sample": _legs(fingerprint=50.0) + [
                {"modality": "face", "sample": b"x", "device_template": object()},
            ],
            "unknown key": [{"modality": "face", "device_score": 1.0, "weights": {"face": 9}}],
        }
        for name, legs in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    verify_multimodal(SUBJECT_MODEL, "s1", legs, actor="agent")
        self.assertEqual(BiometricVerification.objects.count(), 0)

    def test_unregistered_modality_raises_before_any_row(self):
        with self.assertRaises(KeyError):
            verify_multimodal(SUBJECT_MODEL, "s1", _legs(0.5) + [{"modality": "iris", "device_score": 1.0}],
                              actor="agent")
        self.assertEqual(BiometricVerification.objects.count(), 0)

    def test_unknown_or_loose_profile_raises_before_any_row(self):
        BiometricConfig.risk_profiles = {"loose": {"thresholds": {"accept": 0.1}}}
        with self.assertRaises(UnknownRiskProfileError):
            verify_multimodal(SUBJECT_MODEL, "s1", _legs(0.5, 50.0), actor="agent", risk_profile="nope")
        with self.assertRaises(RiskProfileError):
            verify_multimodal(SUBJECT_MODEL, "s1", _legs(0.5, 50.0), actor="agent", risk_profile="loose")
        self.assertEqual(BiometricVerification.objects.count(), 0)


class TestNeverLooserThroughMultimodalVerify(_MultimodalServiceTestCase):
    """The deterministic grid of test_risk_profiles, run through the reachable path."""

    FACE_SCORES = [None, 0.0, 0.3, 0.6, 0.9]
    FINGERPRINT_SCORES = [None, 0.0, 20.0, 40.0, 60.0, 90.0]

    def test_no_profile_ever_loosens_the_decision(self):
        BiometricConfig.modalities = {m: {**cfg, "provider": "device_reported"} for m, cfg in BASE_MODALITIES.items()}
        BiometricConfig.fusion = dict(BASE_FUSION)
        BiometricConfig.risk_profiles = dict(ACCEPTED_PROFILES)

        checked = 0
        for face in self.FACE_SCORES:
            for fingerprint in self.FINGERPRINT_SCORES:
                legs = _legs(face, fingerprint)
                if not legs:
                    continue
                base = verify_multimodal(SUBJECT_MODEL, "s1", legs, actor="agent")
                for name in ACCEPTED_PROFILES:
                    profiled = verify_multimodal(SUBJECT_MODEL, "s1", legs, actor="agent", risk_profile=name)
                    context = (face, fingerprint, name)
                    self.assertLessEqual(
                        _OUTCOME_RANK[profiled.decision.outcome], _OUTCOME_RANK[base.decision.outcome], context,
                    )
                    if profiled.decision.score is not None and base.decision.score is not None:
                        self.assertLessEqual(profiled.decision.score, base.decision.score + 1e-12, context)
                    for base_leg, leg in zip(base.legs, profiled.legs):
                        self.assertGreaterEqual(leg.threshold, base_leg.threshold, context)
                        self.assertLessEqual(leg.verified, base_leg.verified, context)
                    checked += 1
        self.assertEqual(checked, (len(self.FACE_SCORES) * len(self.FINGERPRINT_SCORES) - 1) * len(ACCEPTED_PROFILES))


def _user(perms=None, anonymous=False):
    user = MagicMock()
    user.is_anonymous = anonymous
    user.username = "agent"
    if perms is None:
        user.has_perms.return_value = True
    else:
        # core.models.User.has_perms: an empty list passes, otherwise any one listed right is enough.
        user.has_perms.side_effect = lambda wanted: not wanted or any(p in perms for p in wanted)
    return user


MUTATION = """
mutation {
  verifyBiometricMultimodal(subjectId: "s1", riskProfile: %s, legs: [
    {modality: "face", deviceScore: 0.55},
    {modality: "fingerprint", deviceScore: 40}
  ]) {
    outcome score reasons riskProfile
    legs { modality verified threshold confidence origin riskProfile }
  }
}
"""


class TestMultimodalMutation(_MultimodalTestCase):

    def _execute(self, risk_profile="\"\"", user=None):
        schema = graphene.Schema(query=Query, mutation=Mutation)
        return schema.execute(MUTATION % risk_profile, context_value=SimpleNamespace(user=user or _user(), headers={}))

    def test_accepts_no_rule_arguments(self):
        arguments = set(VerifyBiometricMultimodalMutation.Arguments.__dict__) - {
            "__module__", "__qualname__", "__doc__", "__dict__", "__weakref__",
        }
        self.assertEqual(arguments, {
            "subject_id", "subject_model", "legs", "risk_profile", "fallback", "device_id", "context",
        })

    def test_decision_and_legs_without_profile(self):
        result = self._execute()

        self.assertIsNone(result.errors, result.errors)
        data = result.data["verifyBiometricMultimodal"]
        self.assertEqual(data["outcome"], "accept")
        self.assertAlmostEqual(data["score"], 0.95)
        self.assertEqual(data["riskProfile"], "")
        self.assertEqual([leg["modality"] for leg in data["legs"]], ["face", "fingerprint"])
        self.assertEqual([leg["origin"] for leg in data["legs"]], ["device", "device"])

    def test_profile_applies_over_graphql(self):
        BiometricConfig.risk_profiles = {"strict": {"modality_thresholds": {"face": 1.0}, "floor_decision": "reject"}}

        result = self._execute('"strict"')

        self.assertIsNone(result.errors, result.errors)
        data = result.data["verifyBiometricMultimodal"]
        self.assertEqual(data["outcome"], "review")
        self.assertEqual(data["riskProfile"], "strict")
        self.assertEqual((data["legs"][0]["threshold"], data["legs"][0]["verified"]), (1.0, False))

    def test_unknown_profile_is_an_error_and_writes_nothing(self):
        result = self._execute('"nope"')

        self.assertIsNone(result.data["verifyBiometricMultimodal"])
        self.assertIn("Unknown risk profile 'nope'", str(result.errors[0]))
        self.assertEqual(BiometricVerification.objects.count(), 0)

    def test_needs_the_verify_right(self):
        original = BiometricConfig.gql_biometric_verify_perms
        self.addCleanup(setattr, BiometricConfig, "gql_biometric_verify_perms", original)
        BiometricConfig.gql_biometric_verify_perms = ["174002"]
        for user in (_user(anonymous=True), _user(perms=["174004", "174003"])):
            result = self._execute(user=user)
            self.assertIsInstance(getattr(result.errors[0], "original_error", None), PermissionDenied)
        self.assertEqual(BiometricVerification.objects.count(), 0)
        self.assertIsNone(self._execute(user=_user(perms=["174002"])).errors)

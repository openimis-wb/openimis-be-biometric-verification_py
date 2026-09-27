"""
Unit tests for biometric/risk_profiles.py (docs/wb-biometric-dedup-seam.md §6.8):
validation, the per-key merge, resolve() and the never-loosen guarantee of
fuse() under a profile. No database access.
"""

from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase

from biometric.apps import DEFAULT_CFG, BiometricConfig, _SETTINGS_KEY_MAP
from biometric.risk_profiles import (
    FusionRules,
    RiskProfileError,
    UnknownRiskProfileError,
    base_rules,
    resolve,
    tighten,
    validate_profiles,
    verify_threshold,
)
from biometric.services import _OUTCOME_RANK, fuse

BASE_FUSION = {
    "weights": {"face": 1.0, "fingerprint": 1.0},
    "thresholds": {"accept": 0.7, "review": 0.6},
    "floors": {"fingerprint": 20.0},
    "floor_decision": "review",
}
BASE_MODALITIES = {
    "face": {"provider": "fake_embedding", "threshold": 0.68},
    "fingerprint": {"provider": "fake_matcher", "threshold": 50.0},
}

ACCEPTED_PROFILES = {
    "equal_thresholds": {"thresholds": {"accept": 0.7, "review": 0.6}},
    "raised_accept": {"thresholds": {"accept": 0.8}},
    "raised_fingerprint_floor": {"floors": {"fingerprint": 30.0}},
    "new_face_floor": {"floors": {"face": 0.5}},
    "floor_reject": {"floor_decision": "reject"},
    "fingerprint_required": {"required": ["fingerprint"]},
    "face_threshold_raised": {"modality_thresholds": {"face": 0.8}},
    "modality_thresholds_equal": {"modality_thresholds": {"face": 0.68, "fingerprint": 50.0}},
}


class _RiskProfileConfigTestCase(SimpleTestCase):
    """Saves/restores the BiometricConfig attributes risk profiles read."""

    def setUp(self):
        super().setUp()
        self._snapshot = {
            "risk_profiles": BiometricConfig.risk_profiles,
            "fusion": BiometricConfig.fusion,
            "modalities": BiometricConfig.modalities,
        }
        BiometricConfig.fusion = dict(BASE_FUSION)
        BiometricConfig.modalities = dict(BASE_MODALITIES)
        BiometricConfig.risk_profiles = {}

    def tearDown(self):
        for key, value in self._snapshot.items():
            setattr(BiometricConfig, key, value)
        super().tearDown()


class TestValidateProfiles(_RiskProfileConfigTestCase):

    def test_default_config_has_no_profiles_and_no_errors(self):
        self.assertEqual(DEFAULT_CFG["risk_profiles"], {})
        self.assertEqual(validate_profiles({}), [])

    def test_accepts_profiles_at_least_as_strict(self):
        for name, overrides in ACCEPTED_PROFILES.items():
            with self.subTest(profile=name):
                self.assertEqual(
                    validate_profiles({name: overrides}, fusion=BASE_FUSION, modalities=BASE_MODALITIES), [],
                )

    def test_refuses_looser_or_malformed_profiles(self):
        reject_base = {**BASE_FUSION, "floor_decision": "reject"}
        cases = [
            # (name, overrides, fusion, expected key in every message)
            ("lower_accept", {"thresholds": {"accept": 0.65}}, BASE_FUSION, "accept"),
            ("lower_review", {"thresholds": {"review": 0.5}}, BASE_FUSION, "review"),
            ("review_over_accept", {"thresholds": {"review": 0.75}}, BASE_FUSION, "review"),
            ("lower_floor", {"floors": {"fingerprint": 10.0}}, BASE_FUSION, "fingerprint"),
            ("review_under_reject_base", {"floor_decision": "review"}, reject_base, "floor_decision"),
            ("accept_floor_decision", {"floor_decision": "accept"}, BASE_FUSION, "floor_decision"),
            ("with_weights", {"weights": {"fingerprint": 5.0}}, BASE_FUSION, "weights"),
            ("unknown_key", {"strictness": "high"}, BASE_FUSION, "strictness"),
            ("empty", {}, BASE_FUSION, "empty"),
            ("not_a_dict", "strict", BASE_FUSION, "not_a_dict"),
            ("   ", {"floor_decision": "reject"}, BASE_FUSION, "name"),
            ("x" * 65, {"floor_decision": "reject"}, BASE_FUSION, "name"),
            ("bool_threshold", {"thresholds": {"accept": True}}, BASE_FUSION, "accept"),
            ("nan_threshold", {"thresholds": {"accept": float("nan")}}, BASE_FUSION, "accept"),
            ("negative_floor", {"floors": {"face": -1.0}}, BASE_FUSION, "face"),
            ("lower_face_threshold", {"modality_thresholds": {"face": 0.5}}, BASE_FUSION, "face"),
            ("unconfigured_modality", {"modality_thresholds": {"iris": 0.9}}, BASE_FUSION, "iris"),
            ("required_as_str", {"required": "fingerprint"}, BASE_FUSION, "required"),
        ]
        for name, overrides, fusion, key in cases:
            with self.subTest(profile=name[:20], key=key):
                errors = validate_profiles({name: overrides}, fusion=fusion, modalities=BASE_MODALITIES)
                self.assertTrue(errors, f"{name!r} should be refused")
                for message in errors:
                    self.assertIn(f"Risk profile '{name}'", message)
                self.assertTrue(any(key in message for message in errors), errors)

    def test_non_dict_profiles_is_one_error(self):
        errors = validate_profiles(["strict"], fusion=BASE_FUSION, modalities=BASE_MODALITIES)
        self.assertEqual(len(errors), 1)
        self.assertIn("RISK_PROFILES", errors[0])

    def test_defaults_read_biometric_config(self):
        BiometricConfig.risk_profiles = {"lower_accept": {"thresholds": {"accept": 0.1}}}
        errors = validate_profiles()
        self.assertEqual(len(errors), 1)
        self.assertIn("lower_accept", errors[0])

    def test_base_rules_use_fuse_fallbacks(self):
        rules = base_rules(fusion={}, modalities={"face": {"threshold": 0.68}, "voice": {"provider": "x"}})
        self.assertEqual(rules.thresholds, {"accept": 1.0, "review": 0.0})
        self.assertEqual(rules.floors, {})
        self.assertEqual(rules.floor_decision, "review")
        self.assertEqual(rules.required, frozenset())
        self.assertEqual(rules.modality_thresholds, {"face": 0.68})


class TestTighten(SimpleTestCase):

    def test_per_key_merge_keeps_base_floor_the_profile_omits(self):
        base = FusionRules(
            thresholds={"accept": 0.7, "review": 0.6}, floors={"fingerprint": 20.0}, floor_decision="review",
        )
        merged = tighten(base, {"floors": {"face": 0.5}}, "p")
        self.assertEqual(merged.floors, {"fingerprint": 20.0, "face": 0.5})
        self.assertEqual(merged.thresholds, {"accept": 0.7, "review": 0.6})
        self.assertEqual(merged.risk_profile, "p")

    def test_merge_never_loosens_a_stricter_explicit_base(self):
        base = FusionRules(
            thresholds={"accept": 0.9, "review": 0.6}, floors={"fingerprint": 40.0}, floor_decision="reject",
        )
        merged = tighten(
            base,
            {"thresholds": {"accept": 0.8}, "floor_decision": "review", "floors": {"fingerprint": 30.0}},
            "p",
        )
        self.assertEqual(merged.thresholds["accept"], 0.9)
        self.assertEqual(merged.floor_decision, "reject")
        self.assertEqual(merged.floors["fingerprint"], 40.0)

    def test_required_is_a_union_and_modality_thresholds_take_the_max(self):
        base = FusionRules(
            thresholds={"accept": 0.7, "review": 0.6}, floors={}, floor_decision="review",
            required=frozenset({"face"}), modality_thresholds={"face": 0.68, "fingerprint": 50.0},
        )
        merged = tighten(
            base, {"required": ["fingerprint"], "modality_thresholds": {"face": 0.8, "fingerprint": 40.0}}, "p",
        )
        self.assertEqual(merged.required, frozenset({"face", "fingerprint"}))
        self.assertEqual(merged.modality_thresholds, {"face": 0.8, "fingerprint": 50.0})


class TestResolve(_RiskProfileConfigTestCase):

    def test_unknown_name_raises_UnknownRiskProfileError_listing_configured_names(self):
        BiometricConfig.risk_profiles = {"low": {"floor_decision": "reject"}, "high": {"thresholds": {"accept": 0.9}}}
        base = base_rules(fusion=BASE_FUSION, modalities=BASE_MODALITIES)

        with self.assertRaises(UnknownRiskProfileError) as caught:
            resolve("nope", base)

        self.assertIsInstance(caught.exception, ValueError)
        self.assertIn("'nope'", str(caught.exception))
        self.assertIn("['high', 'low']", str(caught.exception))
        with self.assertRaises(UnknownRiskProfileError):
            verify_threshold("nope", "face", 0.68)

    def test_loose_profile_injected_at_runtime_raises_RiskProfileError(self):
        BiometricConfig.risk_profiles = {"loose": {"thresholds": {"accept": 0.1}}}
        base = base_rules(fusion=BASE_FUSION, modalities=BASE_MODALITIES)

        with self.assertRaises(RiskProfileError) as caught:
            resolve("loose", base)

        self.assertIsInstance(caught.exception, ImproperlyConfigured)
        self.assertIn("loose", str(caught.exception))
        self.assertIn("accept", str(caught.exception))
        with self.assertRaises(RiskProfileError):
            verify_threshold("loose", "face", 0.68)

    def test_resolve_merges_onto_the_base_passed_in(self):
        BiometricConfig.risk_profiles = {"high": {"thresholds": {"accept": 0.8}}}
        caller_base = FusionRules(
            thresholds={"accept": 0.95, "review": 0.6}, floors={}, floor_decision="review",
        )
        self.assertEqual(resolve("high", caller_base).thresholds["accept"], 0.95)

    def test_verify_threshold_takes_the_larger_value(self):
        BiometricConfig.risk_profiles = {"face_08": {"modality_thresholds": {"face": 0.8}}}
        self.assertEqual(verify_threshold("face_08", "face", 0.68), 0.8)
        self.assertEqual(verify_threshold("face_08", "face", 0.9), 0.9)
        self.assertEqual(verify_threshold("face_08", "fingerprint", 50.0), 50.0)


class TestMonotonicity(_RiskProfileConfigTestCase):

    FACE_SCORES = [None, -1.0, -0.3] + [i / 10 for i in range(11)]
    FINGERPRINT_SCORES = [None] + [float(v) for v in range(0, 101, 10)]
    REQUIRED = [frozenset(), frozenset({"fingerprint"})]

    def _assert_never_looser(self, **kwargs):
        checked = 0
        for face in self.FACE_SCORES:
            for fingerprint in self.FINGERPRINT_SCORES:
                scores = {"face": face, "fingerprint": fingerprint}
                for required in self.REQUIRED:
                    base = fuse(scores, required=required, **kwargs)
                    for name in ACCEPTED_PROFILES:
                        profiled = fuse(scores, required=required, risk_profile=name, **kwargs)
                        context = (scores, sorted(required), name, kwargs)
                        self.assertLessEqual(
                            _OUTCOME_RANK[profiled.outcome], _OUTCOME_RANK[base.outcome], context,
                        )
                        if profiled.score is not None and base.score is not None:
                            self.assertLessEqual(profiled.score, base.score + 1e-12, context)
                        checked += 1
        self.assertEqual(
            checked,
            len(self.FACE_SCORES) * len(self.FINGERPRINT_SCORES) * len(self.REQUIRED) * len(ACCEPTED_PROFILES),
        )

    def test_no_profile_ever_loosens_fuse(self):
        BiometricConfig.risk_profiles = dict(ACCEPTED_PROFILES)
        self._assert_never_looser()

    def test_no_profile_ever_loosens_fuse_with_explicit_kwargs(self):
        BiometricConfig.risk_profiles = dict(ACCEPTED_PROFILES)
        self._assert_never_looser(
            weights={"face": 2.0, "fingerprint": 1.0},
            thresholds={"accept": 0.9, "review": 0.5},
            floors={"fingerprint": 40.0},
            floor_decision="reject",
        )


class TestStartupCheck(_RiskProfileConfigTestCase):

    def test_ready_check_logs_each_error_without_raising(self):
        BiometricConfig.risk_profiles = {
            "loose": {"thresholds": {"accept": 0.1}},
            "weighted": {"weights": {"face": 2.0}},
        }

        with self.assertLogs("biometric.apps", "ERROR") as logs:
            BiometricConfig._check_risk_profiles()

        self.assertEqual(len(logs.records), 2)
        self.assertTrue(any("loose" in line and "accept" in line for line in logs.output))
        self.assertTrue(any("weighted" in line and "weights" in line for line in logs.output))

    def test_settings_key_maps_to_risk_profiles(self):
        self.assertEqual(_SETTINGS_KEY_MAP["RISK_PROFILES"], "risk_profiles")
        self.assertTrue(hasattr(BiometricConfig, "risk_profiles"))

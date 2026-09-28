"""
Named risk profiles (docs/wb-biometric-dedup-seam.md §6.8).

A profile is a partial, tighten-only override of the fusion rules in
BIOMETRIC["FUSION"] and of the per-modality thresholds in
BIOMETRIC["MODALITIES"], configured under BIOMETRIC["RISK_PROFILES"]:

    {"high_risk": {"thresholds": {"accept": 0.9},
                   "floors": {"fingerprint": 60},
                   "floor_decision": "reject",
                   "required": ["fingerprint"],
                   "modality_thresholds": {"face": 0.8}}}

Two layers keep a profile from loosening anything: profile_errors() refuses a
declared value looser than the configured base, and tighten() merges key by
key with max / stricter-of / union, so the merged rules are never looser than
the base they are merged onto. 'weights' is not a profile key: weights have
no strictness order, and moving weight between legs can accept a score
vector the base rules review.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

from django.core.exceptions import ImproperlyConfigured

PROFILE_KEYS = frozenset({"thresholds", "floors", "floor_decision", "required", "modality_thresholds"})
THRESHOLD_KEYS = frozenset({"accept", "review"})
# Ordered permissive to strict.
FLOOR_DECISIONS = ("review", "reject")
# Same order as services._OUTCOME_RANK: a lower rank is stricter.
_STRICTNESS = {"reject": 0, "review": 1, "accept": 2}
# Matches BiometricVerification.risk_profile max_length.
MAX_PROFILE_NAME = 64


class RiskProfileError(ImproperlyConfigured):
    """A configured risk profile is malformed or looser than the base rules."""


class UnknownRiskProfileError(ValueError):
    """The caller named a risk profile the configuration does not define."""


@dataclass(frozen=True)
class FusionRules:
    """
    The rules fuse() and verify() apply.

    - thresholds: {"accept": float, "review": float}
    - floors: {modality: minimum raw score}
    - floor_decision: outcome a floor breach caps at
    - required: modalities whose missing score caps the outcome at "review"
    - modality_thresholds: {modality: raw threshold}, on each provider's scale
    - risk_profile: the profile name, "" for the base rules
    """
    thresholds: Dict[str, float]
    floors: Dict[str, float]
    floor_decision: str
    required: FrozenSet[str] = frozenset()
    modality_thresholds: Dict[str, float] = field(default_factory=dict)
    risk_profile: str = ""


def _as_number(value) -> Optional[float]:
    """The value as a float when it is a finite int/float (bool excluded), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def base_rules(*, fusion: dict, modalities: dict) -> FusionRules:
    """The configured base rules, with the same fallbacks fuse() applies."""
    fusion = fusion if isinstance(fusion, dict) else {}
    modalities = modalities if isinstance(modalities, dict) else {}
    thresholds = fusion.get("thresholds", {}) or {}
    return FusionRules(
        thresholds={
            "accept": thresholds.get("accept", 1.0),
            "review": thresholds.get("review", 0.0),
        },
        floors=dict(fusion.get("floors", {}) or {}),
        floor_decision=fusion.get("floor_decision") or "review",
        required=frozenset(),
        modality_thresholds={
            m: float(cfg["threshold"])
            for m, cfg in modalities.items()
            if isinstance(cfg, dict) and _as_number(cfg.get("threshold"))
        },
    )


def _check_name(name, errors, prefix):
    if not isinstance(name, str) or not name.strip():
        errors.append(f"{prefix}the name must be a non-blank string.")
    elif len(name) > MAX_PROFILE_NAME:
        errors.append(f"{prefix}the name is longer than {MAX_PROFILE_NAME} characters.")


def _check_thresholds(value, base, errors, prefix):
    if not isinstance(value, dict) or not value:
        errors.append(f"{prefix}'thresholds' must be a non-empty dict of accept/review.")
        return
    unknown = sorted(str(k) for k in value if k not in THRESHOLD_KEYS)
    if unknown:
        errors.append(f"{prefix}'thresholds' has unknown keys {unknown}; allowed: accept, review.")
    merged = dict(base.thresholds)
    for key in ("accept", "review"):
        if key not in value:
            continue
        number = _as_number(value[key])
        if number is None or number < 0:
            errors.append(f"{prefix}'thresholds.{key}' must be a finite number >= 0, got {value[key]!r}.")
            continue
        base_value = _as_number(base.thresholds.get(key))
        if base_value is not None and number < base_value:
            errors.append(
                f"{prefix}'thresholds.{key}' {number} is below the base {base_value}; "
                "a profile may only raise it."
            )
            continue
        merged[key] = max(number, base_value) if base_value is not None else number
    merged_accept = _as_number(merged.get("accept"))
    merged_review = _as_number(merged.get("review"))
    if merged_accept is not None and merged_review is not None and merged_review > merged_accept:
        errors.append(
            f"{prefix}'thresholds.review' {merged_review} would exceed the merged accept {merged_accept}."
        )


def _check_floors(value, base, errors, prefix):
    if not isinstance(value, dict):
        errors.append(f"{prefix}'floors' must be a dict of modality -> minimum score.")
        return
    for modality, floor in value.items():
        if not isinstance(modality, str) or not modality:
            errors.append(f"{prefix}'floors' keys must be non-empty modality names, got {modality!r}.")
            continue
        number = _as_number(floor)
        if number is None or number < 0:
            errors.append(f"{prefix}'floors.{modality}' must be a finite number >= 0, got {floor!r}.")
            continue
        base_value = _as_number(base.floors.get(modality))
        if base_value is not None and number < base_value:
            errors.append(
                f"{prefix}'floors.{modality}' {number} is below the base floor {base_value}; "
                "a profile may only raise it."
            )


def _check_floor_decision(value, base, errors, prefix):
    if value not in FLOOR_DECISIONS:
        errors.append(f"{prefix}'floor_decision' must be one of {list(FLOOR_DECISIONS)}, got {value!r}.")
        return
    if base.floor_decision in FLOOR_DECISIONS and (
        FLOOR_DECISIONS.index(value) < FLOOR_DECISIONS.index(base.floor_decision)
    ):
        errors.append(
            f"{prefix}'floor_decision' {value!r} is looser than the base {base.floor_decision!r}."
        )


def _check_required(value, errors, prefix):
    if not isinstance(value, (list, tuple)) or not all(isinstance(m, str) and m for m in value):
        errors.append(f"{prefix}'required' must be a list of modality names, got {value!r}.")


def _check_modality_thresholds(value, base, errors, prefix):
    if not isinstance(value, dict):
        errors.append(f"{prefix}'modality_thresholds' must be a dict of modality -> threshold.")
        return
    for modality, threshold in value.items():
        if modality not in base.modality_thresholds:
            errors.append(
                f"{prefix}'modality_thresholds.{modality}' has no configured threshold to tighten."
            )
            continue
        number = _as_number(threshold)
        if number is None:
            errors.append(
                f"{prefix}'modality_thresholds.{modality}' must be a finite number, got {threshold!r}."
            )
            continue
        base_value = base.modality_thresholds[modality]
        if number < base_value:
            errors.append(
                f"{prefix}'modality_thresholds.{modality}' {number} is below the base {base_value}; "
                "a profile may only raise it."
            )


def profile_errors(name, overrides, *, base: FusionRules) -> List[str]:
    """Every reason the profile is malformed or looser than base; [] when it is usable."""
    prefix = f"Risk profile '{name}': "
    errors: List[str] = []
    _check_name(name, errors, prefix)

    if not isinstance(overrides, dict) or not overrides:
        errors.append(f"{prefix}must be a non-empty dict of overrides.")
        return errors

    for key in overrides:
        if key == "weights":
            errors.append(
                f"{prefix}'weights' is not overridable: weights have no strictness order, "
                "so a profile could not be proven not to loosen the base."
            )
        elif key not in PROFILE_KEYS:
            errors.append(f"{prefix}unknown key {key!r}; allowed: {sorted(PROFILE_KEYS)}.")

    if "thresholds" in overrides:
        _check_thresholds(overrides["thresholds"], base, errors, prefix)
    if "floors" in overrides:
        _check_floors(overrides["floors"], base, errors, prefix)
    if "floor_decision" in overrides:
        _check_floor_decision(overrides["floor_decision"], base, errors, prefix)
    if "required" in overrides:
        _check_required(overrides["required"], errors, prefix)
    if "modality_thresholds" in overrides:
        _check_modality_thresholds(overrides["modality_thresholds"], base, errors, prefix)
    return errors


def validate_profiles(profiles=None, *, fusion=None, modalities=None) -> List[str]:
    """
    Every error of every profile, checked against the base built from fusion
    and modalities. Each argument defaults to the BiometricConfig attribute.
    Never raises.
    """
    from .apps import BiometricConfig

    profiles = BiometricConfig.risk_profiles if profiles is None else profiles
    fusion = BiometricConfig.fusion if fusion is None else fusion
    modalities = BiometricConfig.modalities if modalities is None else modalities

    if not isinstance(profiles, dict):
        return [f"BIOMETRIC['RISK_PROFILES'] must be a dict of name -> overrides, got {type(profiles).__name__}."]

    base = base_rules(fusion=fusion, modalities=modalities)
    errors: List[str] = []
    for name, overrides in profiles.items():
        errors.extend(profile_errors(name, overrides, base=base))
    return errors


def _max_declared(base_value, profile_value):
    """The larger of the two; the other one when either is None."""
    if profile_value is None:
        return base_value
    if base_value is None:
        return profile_value
    return max(base_value, profile_value)


def _stricter_decision(base_value, profile_value):
    """The stricter floor decision; an unrecognised base yields to a declared profile value."""
    if profile_value is None:
        return base_value
    return min((base_value, profile_value), key=lambda v: _STRICTNESS.get(v, len(_STRICTNESS)))


def tighten(base: FusionRules, overrides: dict, name: str) -> FusionRules:
    """
    base with the profile merged per key: max for accept, review, each floor
    and each modality threshold; the stricter floor_decision; the union of
    required. Keys the profile omits keep the base value, so the result is
    never looser than base.
    """
    profile_thresholds = overrides.get("thresholds") or {}
    thresholds = {
        key: _max_declared(base.thresholds.get(key), profile_thresholds.get(key))
        for key in ("accept", "review")
    }

    floors = dict(base.floors)
    for modality, floor in (overrides.get("floors") or {}).items():
        floors[modality] = _max_declared(base.floors.get(modality), floor)

    modality_thresholds = dict(base.modality_thresholds)
    for modality, threshold in (overrides.get("modality_thresholds") or {}).items():
        modality_thresholds[modality] = _max_declared(base.modality_thresholds.get(modality), threshold)

    return FusionRules(
        thresholds=thresholds,
        floors=floors,
        floor_decision=_stricter_decision(base.floor_decision, overrides.get("floor_decision")),
        required=frozenset(base.required) | frozenset(overrides.get("required") or ()),
        modality_thresholds=modality_thresholds,
        risk_profile=name,
    )


def _configured_profile(risk_profile):
    """The overrides of a configured, valid profile; raises otherwise."""
    from .apps import BiometricConfig

    profiles = BiometricConfig.risk_profiles or {}
    if not isinstance(profiles, dict):
        raise RiskProfileError(
            f"BIOMETRIC['RISK_PROFILES'] must be a dict of name -> overrides, got {type(profiles).__name__}."
        )
    if not isinstance(risk_profile, str) or risk_profile not in profiles:
        raise UnknownRiskProfileError(
            f"Unknown risk profile {risk_profile!r}; configured: {sorted(profiles)}"
        )
    overrides = profiles[risk_profile]
    errors = profile_errors(
        risk_profile, overrides,
        base=base_rules(fusion=BiometricConfig.fusion, modalities=BiometricConfig.modalities),
    )
    if errors:
        raise RiskProfileError("; ".join(errors))
    return overrides


def resolve(risk_profile: str, base: FusionRules) -> FusionRules:
    """
    The named profile merged onto base. Validation runs against the configured
    base; the merge runs against the base passed in.

    Raises UnknownRiskProfileError for a name the configuration does not define
    and RiskProfileError for a profile that is malformed or looser than the
    configured base.
    """
    return tighten(base, _configured_profile(risk_profile), risk_profile)


def verify_threshold(risk_profile: str, modality: str, base_threshold: float) -> float:
    """
    The threshold verify() applies under the named profile: the larger of
    base_threshold and the profile's modality threshold. Raises as resolve().
    """
    overrides = _configured_profile(risk_profile)
    profile_value = (overrides.get("modality_thresholds") or {}).get(modality)
    if profile_value is None:
        return base_threshold
    return max(base_threshold, profile_value)


def _numbers_by_modality(value) -> Dict[str, Optional[float]]:
    """{modality: number or None} from a dict; non-string keys and non-dict values give {}."""
    if not isinstance(value, dict):
        return {}
    return {m: _as_number(v) for m, v in value.items() if isinstance(m, str)}


def _declared_overrides(overrides) -> dict:
    """
    The profile keys a screen may show, each value reduced to the type it
    must have: numbers (None when not a finite number), a floor decision among
    FLOOR_DECISIONS (else None), modality-name strings. Other keys are dropped.
    """
    if not isinstance(overrides, dict):
        return {}
    declared = {}
    thresholds = overrides.get("thresholds")
    if isinstance(thresholds, dict):
        declared["thresholds"] = {k: _as_number(thresholds[k]) for k in ("accept", "review") if k in thresholds}
    if "floors" in overrides:
        declared["floors"] = _numbers_by_modality(overrides["floors"])
    if "floor_decision" in overrides:
        value = overrides["floor_decision"]
        declared["floor_decision"] = value if value in FLOOR_DECISIONS else None
    if "required" in overrides:
        value = overrides["required"]
        declared["required"] = [m for m in value if isinstance(m, str)] if isinstance(value, (list, tuple)) else []
    if "modality_thresholds" in overrides:
        declared["modality_thresholds"] = _numbers_by_modality(overrides["modality_thresholds"])
    return declared


def _rules_dict(rules: FusionRules, weights: Dict[str, Optional[float]]) -> dict:
    return {
        "thresholds": {k: _as_number(rules.thresholds.get(k)) for k in ("accept", "review")},
        "floors": _numbers_by_modality(rules.floors),
        "floor_decision": rules.floor_decision if isinstance(rules.floor_decision, str) else None,
        "required": sorted(m for m in rules.required if isinstance(m, str)),
        "modality_thresholds": _numbers_by_modality(rules.modality_thresholds),
        "weights": dict(weights),
    }


def decision_criteria() -> dict:
    """
    The configured decision rules for an admin screen, restricted to the
    fusion and profile keys: {"base": rules, "profiles": [...]} with each
    profile as {"name", "valid", "errors", "overrides", "effective"} sorted by
    name. effective is the profile merged onto the base, None for an invalid
    profile. No other configuration value (provider settings, keys) is read.
    """
    from .apps import BiometricConfig

    fusion = BiometricConfig.fusion if isinstance(BiometricConfig.fusion, dict) else {}
    base = base_rules(fusion=fusion, modalities=BiometricConfig.modalities)
    weights = _numbers_by_modality(fusion.get("weights") or {})

    profiles = BiometricConfig.risk_profiles if isinstance(BiometricConfig.risk_profiles, dict) else {}
    described = []
    for name in sorted(profiles, key=str):
        overrides = profiles[name]
        errors = profile_errors(name, overrides, base=base)
        described.append({
            "name": str(name),
            "valid": not errors,
            "errors": errors,
            "overrides": _declared_overrides(overrides),
            "effective": None if errors else _rules_dict(tighten(base, overrides, name), weights),
        })
    return {"base": _rules_dict(base, weights), "profiles": described}

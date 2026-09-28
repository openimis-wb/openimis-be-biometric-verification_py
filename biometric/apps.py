import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)

MODULE_NAME = "biometric"

DEFAULT_CFG = {
    "subject_model": "individual.Individual",
    "modalities": {
        "face": {"provider": "deepface", "threshold": 0.32},
        "fingerprint": {"provider": "device_reported", "threshold": 48},
    },
    "vector_index": "numpy",  # "numpy" | "pgvector" (pgvector requires the biometric_pgvector app)
    "template_key": None,     # Fernet key; templates/vectors encrypted at rest when set
    "require_consent": False,
    "dedup_threshold": {"face": 0.62},  # similarity at/above which a candidate is emitted
    "fusion": {
        "weights": {"face": 1.0},
        "thresholds": {"accept": 0.7, "review": 0.6},
        "floors": {},
        "floor_decision": "review",
    },
    # Enrolment quality gate (biometric/quality.py): "advisory" stores the
    # verdict only, "enforce" refuses a REFUSED sample. Nested keys are lowercase.
    "quality": {
        "mode": "advisory",
        "modalities": {
            "face": {
                "min_sharpness": 100.0,
                "max_yaw": 20.0,
                "max_pitch": None,
                "max_roll": 20.0,
                "max_yaw_ratio": None,
                "min_lower_face_uniformity": None,
                "min_quality": None,
            },
        },
    },
    # Named tighten-only overrides of FUSION / MODALITIES thresholds
    # (biometric/risk_profiles.py).
    "risk_profiles": {},
    # 1:N impersonation probe inside verify() (biometric/impersonation.py).
    # Nested keys are lowercase; missing keys take PROBE_DEFAULTS at read time.
    "impersonation_probe": {
        "enabled": False, "modalities": ["face"], "top_k": 5, "thresholds": {}, "margin": None, "device_path": False,
    },
    # Hash-chained audit events and alert rules (biometric/audit_chain.py,
    # biometric/audit_rules.py). Nested keys are lowercase; "rules" overrides
    # audit_rules.DEFAULT_RULES per kind, key by key.
    "audit": {"enabled": False, "rules": {}},
    "gql_biometric_enrol_perms": ["174001"],
    "gql_biometric_verify_perms": ["174002"],
    "gql_biometric_identify_perms": ["174003"],
    "gql_biometric_read_perms": ["174004"],
    "gql_biometric_audit_perms": ["174005"],   # read audit events and alerts
    "gql_biometric_alert_perms": ["174006"],   # acknowledge and resolve alerts
}

# Maps uppercase Django settings keys -> lowercase ModuleConfiguration keys.
_SETTINGS_KEY_MAP = {
    "SUBJECT_MODEL": "subject_model",
    "MODALITIES": "modalities",
    "VECTOR_INDEX": "vector_index",
    "TEMPLATE_KEY": "template_key",
    "REQUIRE_CONSENT": "require_consent",
    "DEDUP_THRESHOLD": "dedup_threshold",
    "FUSION": "fusion",
    "QUALITY": "quality",
    "RISK_PROFILES": "risk_profiles",
    "IMPERSONATION_PROBE": "impersonation_probe",
    "AUDIT": "audit",
    "GQL_BIOMETRIC_ENROL_PERMS": "gql_biometric_enrol_perms",
    "GQL_BIOMETRIC_VERIFY_PERMS": "gql_biometric_verify_perms",
    "GQL_BIOMETRIC_IDENTIFY_PERMS": "gql_biometric_identify_perms",
    "GQL_BIOMETRIC_READ_PERMS": "gql_biometric_read_perms",
    "GQL_BIOMETRIC_AUDIT_PERMS": "gql_biometric_audit_perms",
    "GQL_BIOMETRIC_ALERT_PERMS": "gql_biometric_alert_perms",
}


class BiometricConfig(AppConfig):
    name = MODULE_NAME

    subject_model = "individual.Individual"
    modalities = {
        "face": {"provider": "deepface", "threshold": 0.32},
        "fingerprint": {"provider": "device_reported", "threshold": 48},
    }
    vector_index = "numpy"
    template_key = None
    require_consent = False
    dedup_threshold = {"face": 0.62}
    fusion = {
        "weights": {"face": 1.0},
        "thresholds": {"accept": 0.7, "review": 0.6},
        "floors": {},
        "floor_decision": "review",
    }
    quality = {
        "mode": "advisory",
        "modalities": {
            "face": {
                "min_sharpness": 100.0,
                "max_yaw": 20.0,
                "max_pitch": None,
                "max_roll": 20.0,
                "max_yaw_ratio": None,
                "min_lower_face_uniformity": None,
                "min_quality": None,
            },
        },
    }
    # name -> partial override of thresholds / floors / floor_decision /
    # required / modality_thresholds; each value may only equal or tighten
    # the base in fusion and modalities.
    risk_profiles = {}
    impersonation_probe = {
        "enabled": False, "modalities": ["face"], "top_k": 5, "thresholds": {}, "margin": None, "device_path": False,
    }
    audit = {"enabled": False, "rules": {}}
    gql_biometric_enrol_perms = ["174001"]
    gql_biometric_verify_perms = ["174002"]
    gql_biometric_identify_perms = ["174003"]
    gql_biometric_read_perms = ["174004"]
    gql_biometric_audit_perms = ["174005"]
    gql_biometric_alert_perms = ["174006"]

    def __load_config(self, cfg):
        for field, value in cfg.items():
            if hasattr(BiometricConfig, field):
                setattr(BiometricConfig, field, value)

    def ready(self):
        from core.models import ModuleConfiguration
        from django.conf import settings

        # 1. Load from DB / defaults (standard openIMIS pattern).
        cfg = ModuleConfiguration.get_or_default(MODULE_NAME, DEFAULT_CFG)

        # 2. django.conf.settings.BIOMETRIC overrides DB config.
        #    Accepts both UPPER_CASE and lower_case keys.
        for key, value in getattr(settings, "BIOMETRIC", {}).items():
            normalised = _SETTINGS_KEY_MAP.get(key.upper(), key.lower())
            cfg[normalised] = value

        self.__load_config(cfg)
        self._check_quality_config()
        self._check_risk_profiles()
        self._check_audit_config()

        # 3. Bind the deduplication.subject_merged -> consolidate() signal.
        from . import signals  # noqa: F401

        # 4. Register this module's candidate source with the deduplication
        #    module, if installed (docs/wb-biometric-dedup-seam.md §2.1).
        self._register_candidate_source()

    @staticmethod
    def _check_quality_config():
        """Logs an unusable quality gate config; enrol() raises on it through quality.mode()."""
        import logging

        from .quality import MODES, _warn_if_pillow_missing_in_enforce

        quality = BiometricConfig.quality if isinstance(BiometricConfig.quality, dict) else {}
        if quality.get("mode", "advisory") not in MODES:
            logging.getLogger(__name__).error(
                "BIOMETRIC['QUALITY']['mode'] is %r; expected one of %s. enrol() refuses to run until it is fixed.",
                quality.get("mode"), MODES,
            )
        _warn_if_pillow_missing_in_enforce()

    @staticmethod
    def _check_risk_profiles():
        """Logs every invalid risk profile; risk_profiles.resolve() raises when one is used."""
        from .risk_profiles import validate_profiles

        for message in validate_profiles():
            logger.error("BIOMETRIC['RISK_PROFILES']: %s", message)

    @staticmethod
    def _check_audit_config():
        """Logs every problem in BIOMETRIC['AUDIT']; audit_rules.rule_params() raises when a bad rule is read."""
        from .audit_rules import validate_audit_config

        for message in validate_audit_config(BiometricConfig.audit):
            logger.error("BIOMETRIC['AUDIT']: %s", message)

    @staticmethod
    def _register_candidate_source():
        try:
            from deduplication.sources import register
        except ImportError:
            return
        from .dedup_source import BiometricCandidateSource
        register(BiometricCandidateSource())

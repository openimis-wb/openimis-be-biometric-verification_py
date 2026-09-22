from django.apps import AppConfig

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
    "gql_biometric_enrol_perms": ["174001"],
    "gql_biometric_verify_perms": ["174002"],
    "gql_biometric_identify_perms": ["174003"],
    "gql_biometric_read_perms": ["174004"],
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
    "GQL_BIOMETRIC_ENROL_PERMS": "gql_biometric_enrol_perms",
    "GQL_BIOMETRIC_VERIFY_PERMS": "gql_biometric_verify_perms",
    "GQL_BIOMETRIC_IDENTIFY_PERMS": "gql_biometric_identify_perms",
    "GQL_BIOMETRIC_READ_PERMS": "gql_biometric_read_perms",
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
    gql_biometric_enrol_perms = ["174001"]
    gql_biometric_verify_perms = ["174002"]
    gql_biometric_identify_perms = ["174003"]
    gql_biometric_read_perms = ["174004"]

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

        # 3. Bind the deduplication.subject_merged -> consolidate() signal.
        from . import signals  # noqa: F401

        # 4. Register this module's candidate source with the deduplication
        #    module, if installed (docs/wb-biometric-dedup-seam.md §2.1).
        self._register_candidate_source()

    @staticmethod
    def _register_candidate_source():
        try:
            from deduplication.sources import register
        except ImportError:
            return
        from .dedup_source import BiometricCandidateSource
        register(BiometricCandidateSource())

from django.apps import AppConfig

MODULE_NAME = "biometric_verification"

DEFAULT_CFG = {
    "provider": "deepface",
    "provider_config": {
        "model_name": "ArcFace",
        "detector_backend": "retinaface",  # Changed from opencv - more accurate face detection
        "enforce_detection": True,          # CRITICAL: Ensure face is detected
    },
    "store_embeddings": True,
    "similarity_threshold": 0.68,
    "max_image_size_px": 1024,
    "gql_mutation_verify_face_perms": [],
    "gql_mutation_compute_embedding_perms": [],
    # WebSocket streaming settings
    "sampling_interval_seconds": 5,  # Time between verifications (5 sec = 12 verifs/min)
    "websocket_auth_tokens": [],     # Optional list of auth tokens; empty = public access

    # --- Multimodal identity + deduplication seam (docs/wb-biometric-dedup-seam.md §3.2) ---
    "subject_model": "individual.Individual",
    "modalities": {
        # 0.32 is 1.0 - similarity_threshold (0.68): the legacy insuree flow's
        # 0.68 is a cosine DISTANCE cutoff (verified when distance <= 0.68);
        # this new path scores SIMILARITY (verified when similarity >= threshold),
        # so the same operating point is 1.0 - 0.68 = 0.32 here.
        "face": {"provider": "deepface", "threshold": 0.32},
        "fingerprint": {"provider": "device_reported", "threshold": 48},
    },
    "vector_index": "numpy",  # "numpy" | "pgvector" (pgvector only if importable)
    "template_key": None,     # Fernet key; templates/vectors encrypted at rest when set
    "require_consent": False,
    "dedup_threshold": {"face": 0.62},  # similarity (not distance) at/above which a candidate is emitted
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

# Maps uppercase Django settings keys → lowercase ModuleConfiguration keys.
# Allows operators to configure via settings.BIOMETRIC_VERIFICATION using either
# style: PROVIDER / provider, STORE_EMBEDDINGS / store_embeddings, etc.
_SETTINGS_KEY_MAP = {
    "PROVIDER": "provider",
    "PROVIDER_CONFIG": "provider_config",
    "STORE_EMBEDDINGS": "store_embeddings",
    "SIMILARITY_THRESHOLD": "similarity_threshold",
    "MAX_IMAGE_SIZE_PX": "max_image_size_px",
    "GQL_MUTATION_VERIFY_FACE_PERMS": "gql_mutation_verify_face_perms",
    "GQL_MUTATION_COMPUTE_EMBEDDING_PERMS": "gql_mutation_compute_embedding_perms",
    "SAMPLING_INTERVAL_SECONDS": "sampling_interval_seconds",
    "WEBSOCKET_AUTH_TOKENS": "websocket_auth_tokens",
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


# Mutations that are publicly accessible without a JWT token.
# Added to settings.GRAPHQL_JWT["JWT_ALLOW_ANY_CLASSES"] at app startup.
_PUBLIC_MUTATIONS = [
    "biometric_verification.schema.VerifyFaceMutation",
]


def _whitelist_public_mutations(settings):
    """Append public mutation classes to graphql_jwt's allow-any list."""
    jwt_settings = getattr(settings, "GRAPHQL_JWT", {})
    allow_any = jwt_settings.get("JWT_ALLOW_ANY_CLASSES", [])
    for cls_path in _PUBLIC_MUTATIONS:
        if cls_path not in allow_any:
            allow_any.append(cls_path)
    jwt_settings["JWT_ALLOW_ANY_CLASSES"] = allow_any
    settings.GRAPHQL_JWT = jwt_settings


class BiometricVerificationConfig(AppConfig):
    name = MODULE_NAME

    provider = None
    provider_config = {}
    store_embeddings = True
    similarity_threshold = 0.68
    max_image_size_px = 1024
    gql_mutation_verify_face_perms = []
    gql_mutation_compute_embedding_perms = []
    # WebSocket streaming settings
    sampling_interval_seconds = 5
    websocket_auth_tokens = []

    # Multimodal identity + deduplication seam
    subject_model = "individual.Individual"
    modalities = {
        # See DEFAULT_CFG above: 0.32 = 1.0 - similarity_threshold (a distance cutoff).
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
            if hasattr(BiometricVerificationConfig, field):
                setattr(BiometricVerificationConfig, field, value)

    def ready(self):
        from core.models import ModuleConfiguration
        from django.conf import settings

        # 1. Load from DB / defaults (standard openIMIS pattern)
        cfg = ModuleConfiguration.get_or_default(MODULE_NAME, DEFAULT_CFG)

        # 2. django.conf.settings.BIOMETRIC_VERIFICATION overrides DB config.
        #    Accepts both UPPER_CASE and lower_case keys.
        for key, value in getattr(settings, "BIOMETRIC_VERIFICATION", {}).items():
            normalised = _SETTINGS_KEY_MAP.get(key.upper(), key.lower())
            cfg[normalised] = value

        self.__load_config(cfg)

        # 3. Whitelist verifyFace so it can be called without a JWT.
        #    This enables the public kiosk page (see views.py / SECURITY.md).
        #    ⚠️  Read SECURITY.md before deploying to production.
        _whitelist_public_mutations(settings)

        # 4. Register Django signals for automatic claim risk score updates
        from . import signals  # noqa: F401

        # 5. Register this module's candidate source with the deduplication
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

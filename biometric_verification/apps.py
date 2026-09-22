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

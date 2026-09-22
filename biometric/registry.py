import logging
from typing import Dict, Tuple, Type

from .providers.base import ModalityProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Central registry mapping (modality, provider name) to ModalityProvider
    classes. Instances are cached per modality (one active provider per
    modality at a time, per BIOMETRIC["MODALITIES"] configuration).
    """

    _modality_registry: Dict[Tuple[str, str], Type[ModalityProvider]] = {}
    _modality_instances: Dict[str, ModalityProvider] = {}

    @classmethod
    def register_modality(cls, modality: str, name: str, provider_class: Type[ModalityProvider]) -> None:
        """Register a ModalityProvider class under (modality, name)."""
        if not (isinstance(provider_class, type) and issubclass(provider_class, ModalityProvider)):
            raise TypeError(f"{provider_class} must extend ModalityProvider.")
        cls._modality_registry[(modality, name)] = provider_class
        cls._modality_instances.pop(modality, None)
        logger.debug(
            "Registered modality provider: %s/%s -> %s", modality, name, provider_class.__name__
        )

    @classmethod
    def get_provider(cls, modality: str) -> ModalityProvider:
        """
        Return a (cached) ModalityProvider instance for the given modality,
        configured via BIOMETRIC["MODALITIES"][modality].
        """
        if modality in cls._modality_instances:
            return cls._modality_instances[modality]

        from .apps import BiometricConfig

        modality_cfg = dict(BiometricConfig.modalities.get(modality, {}))
        name = modality_cfg.pop("provider", None)
        threshold = modality_cfg.pop("threshold", None)
        if name is None:
            raise KeyError(
                f"No provider configured for modality '{modality}'. "
                f"Set BIOMETRIC['MODALITIES']['{modality}']['provider']."
            )

        key = (modality, name)
        if key not in cls._modality_registry:
            available = list(cls._modality_registry.keys())
            raise KeyError(
                f"Modality provider '{name}' is not registered for modality '{modality}'. "
                f"Available: {available}."
            )

        provider_class = cls._modality_registry[key]
        kwargs = dict(modality_cfg)
        kwargs.setdefault("modality", modality)
        if threshold is not None:
            kwargs.setdefault("default_threshold", threshold)

        instance = provider_class(**kwargs)
        cls._modality_instances[modality] = instance
        logger.info(
            "Instantiated modality provider '%s' for modality '%s' (%s)",
            name, modality, provider_class.__name__,
        )
        return instance


# ---------------------------------------------------------------------------
# Register built-in providers. Deep imports are deferred so that a missing
# optional dependency (deepface) only raises an error when actually used.
# ---------------------------------------------------------------------------

def _register_builtin_providers() -> None:
    try:
        from .providers.deepface_provider import DeepFaceProvider
        ProviderRegistry.register_modality("face", "deepface", DeepFaceProvider)
    except ImportError:
        logger.debug(
            "deepface provider not registered: deepface package is not installed. "
            "Install with: pip install 'openimis-be-biometric_verification[deepface]'"
        )

    from .providers.device_reported import DeviceReportedMatcher
    # device_reported matches on the device for any modality — pre-register it
    # for every documented modality so configuring it needs no code change.
    for _modality in ("face", "fingerprint", "voice", "iris", "palmvein"):
        ProviderRegistry.register_modality(_modality, "device_reported", DeviceReportedMatcher)


_register_builtin_providers()

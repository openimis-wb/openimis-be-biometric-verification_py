import logging
from typing import Dict, Tuple, Type

from .providers.base import BaseBiometricProvider, ModalityProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Central registry that maps provider name strings to provider classes.

    Built-in providers are registered at the bottom of this file.
    External / licensed providers register themselves by calling
    ProviderRegistry.register() from their own module.

    Usage
    -----
    # Register a custom provider
    ProviderRegistry.register("my_provider", MyProvider)

    # Retrieve the active provider (reads BiometricVerificationConfig.provider)
    provider = ProviderRegistry.get_active_provider()
    """

    _registry: Dict[str, Type[BaseBiometricProvider]] = {}
    _instances: Dict[str, BaseBiometricProvider] = {}

    # Multimodal registry — keyed separately so the legacy face-only registry
    # above (and its tests, which clear _registry/_instances directly) is untouched.
    _modality_registry: Dict[Tuple[str, str], Type[ModalityProvider]] = {}
    _modality_instances: Dict[str, ModalityProvider] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseBiometricProvider]) -> None:
        """Register a provider class under the given name."""
        if not issubclass(provider_class, BaseBiometricProvider):
            raise TypeError(
                f"{provider_class} must extend BaseBiometricProvider."
            )
        cls._registry[name] = provider_class
        # Invalidate any cached instance for this name so the next call to
        # get_active_provider() creates a fresh one.
        cls._instances.pop(name, None)
        logger.debug("Registered biometric provider: %s → %s", name, provider_class.__name__)

    @classmethod
    def get_active_provider(cls) -> BaseBiometricProvider:
        """
        Return a (cached) instance of the provider named in
        BiometricVerificationConfig.provider.

        The instance is created once per provider name and reused on
        subsequent calls — this avoids reloading heavy ML models on
        every request.

        Raises
        ------
        KeyError   if the configured provider name is not registered.
        TypeError  if the registered class is not a BaseBiometricProvider.
        """
        from .apps import BiometricVerificationConfig

        name = BiometricVerificationConfig.provider or "deepface"

        if name not in cls._instances:
            if name not in cls._registry:
                available = list(cls._registry.keys())
                raise KeyError(
                    f"Biometric provider '{name}' is not registered. "
                    f"Available providers: {available}. "
                    f"Check BIOMETRIC_VERIFICATION['PROVIDER'] in your settings."
                )
            provider_class = cls._registry[name]
            config = BiometricVerificationConfig.provider_config or {}
            cls._instances[name] = provider_class(**config)
            logger.info(
                "Instantiated biometric provider '%s' (%s)",
                name,
                provider_class.__name__,
            )

        return cls._instances[name]

    @classmethod
    def available_providers(cls) -> list:
        """Return the list of registered provider names."""
        return list(cls._registry.keys())

    # ------------------------------------------------------------------
    # Multimodal registry (docs/wb-biometric-dedup-seam.md §3.2)
    # ------------------------------------------------------------------

    @classmethod
    def register_modality(cls, modality: str, name: str, provider_class: Type[ModalityProvider]) -> None:
        """Register a ModalityProvider class under (modality, name)."""
        if not (isinstance(provider_class, type) and issubclass(provider_class, ModalityProvider)):
            raise TypeError(f"{provider_class} must extend ModalityProvider.")
        cls._modality_registry[(modality, name)] = provider_class
        cls._modality_instances.pop(modality, None)
        logger.debug(
            "Registered modality provider: %s/%s → %s", modality, name, provider_class.__name__
        )

    @classmethod
    def get_provider(cls, modality: str) -> ModalityProvider:
        """
        Return a (cached) ModalityProvider instance for the given modality,
        configured via BIOMETRIC_VERIFICATION["MODALITIES"][modality].
        """
        if modality in cls._modality_instances:
            return cls._modality_instances[modality]

        from .apps import BiometricVerificationConfig

        modality_cfg = dict(BiometricVerificationConfig.modalities.get(modality, {}))
        name = modality_cfg.pop("provider", None)
        threshold = modality_cfg.pop("threshold", None)
        if name is None:
            raise KeyError(
                f"No provider configured for modality '{modality}'. "
                f"Set BIOMETRIC_VERIFICATION['MODALITIES']['{modality}']['provider']."
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
# Register built-in providers
# Deep imports are deferred so that missing optional deps (deepface, boto3…)
# only raise an error when the provider is actually used, not at import time.
# ---------------------------------------------------------------------------

def _register_builtin_providers() -> None:
    try:
        from .providers.deepface_provider import DeepFaceProvider
        ProviderRegistry.register("deepface", DeepFaceProvider)
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

import logging
from typing import Dict, Type

from .providers.base import BaseBiometricProvider

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


# ---------------------------------------------------------------------------
# Register built-in providers
# Deep imports are deferred so that missing optional deps (deepface, boto3…)
# only raise an error when the provider is actually used, not at import time.
# ---------------------------------------------------------------------------

def _register_builtin_providers() -> None:
    try:
        from .providers.deepface_provider import DeepFaceProvider
        ProviderRegistry.register("deepface", DeepFaceProvider)
    except ImportError:
        logger.debug(
            "deepface provider not registered: deepface package is not installed. "
            "Install with: pip install 'openimis-be-biometric_verification[deepface]'"
        )


_register_builtin_providers()

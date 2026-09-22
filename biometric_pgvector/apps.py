from django.apps import AppConfig
from django.core.exceptions import ImproperlyConfigured

MODULE_NAME = "biometric_pgvector"


class BiometricPgvectorConfig(AppConfig):
    name = MODULE_NAME
    label = MODULE_NAME

    def ready(self):
        self._guard_plaintext_index()
        from . import receivers

        receivers.connect()

    @staticmethod
    def _guard_plaintext_index():
        """
        Refuse to start when BIOMETRIC['TEMPLATE_KEY'] is set (templates are
        encrypted at rest) unless BIOMETRIC['ALLOW_PLAINTEXT_INDEX'] is True —
        an ANN index cannot search encrypted vectors, so this app always
        stores them in clear (docs/wb-biometric-dedup-seam.md §6.2).

        Reads biometric.apps.BiometricConfig.template_key rather than the raw
        config: every manifest lists "biometric" before "biometric_pgvector"
        (it is this app's own FK target), Django calls AppConfig.ready() in
        INSTALLED_APPS order, so BiometricConfig.ready() has already loaded
        template_key by the time this runs.
        """
        from biometric.apps import BiometricConfig

        from .config import allow_plaintext_index

        if BiometricConfig.template_key is not None and not allow_plaintext_index():
            raise ImproperlyConfigured(
                "biometric_pgvector stores vectors in clear for ANN search, but "
                "BIOMETRIC['TEMPLATE_KEY'] is set (templates are encrypted at "
                "rest). Set BIOMETRIC['ALLOW_PLAINTEXT_INDEX'] = True to accept "
                "plaintext vectors in this index, or unset TEMPLATE_KEY / "
                "uninstall biometric_pgvector."
            )

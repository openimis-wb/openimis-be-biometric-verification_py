"""
Reads the BIOMETRIC config keys this app depends on — ALLOW_PLAINTEXT_INDEX,
HNSW_EF_SEARCH and TEMPLATE_KEY (docs/wb-biometric-dedup-seam.md §6.2). Same source and override order as BiometricConfig.ready() (the
"biometric" ModuleConfiguration row, then django.conf.settings.BIOMETRIC),
replicated here rather than added to biometric/apps.py.
"""

from core.models import ModuleConfiguration
from django.conf import settings


def _raw_biometric_cfg():
    cfg = dict(ModuleConfiguration.get_or_default("biometric", {}))
    for key, value in (getattr(settings, "BIOMETRIC", {}) or {}).items():
        cfg[key.lower()] = value
    return cfg


def template_key():
    """The encryption key, read from the configuration source itself rather than
    from BiometricConfig, so the answer does not depend on which app's ready()
    ran first."""
    return _raw_biometric_cfg().get("template_key")


def allow_plaintext_index():
    return bool(_raw_biometric_cfg().get("allow_plaintext_index", False))


def hnsw_ef_search():
    """Read once by BiometricPgvectorConfig.ready(); identify reads the stored value."""
    return int(_raw_biometric_cfg().get("hnsw_ef_search", 200))

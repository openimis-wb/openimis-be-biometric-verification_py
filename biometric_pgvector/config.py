"""
Reads the BIOMETRIC config keys that belong to this app but that
biometric.apps.BiometricConfig does not register in its DEFAULT_CFG —
ALLOW_PLAINTEXT_INDEX and HNSW_EF_SEARCH (docs/wb-biometric-dedup-seam.md
§6.2). Same source and override order as BiometricConfig.ready() (the
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


def allow_plaintext_index():
    return bool(_raw_biometric_cfg().get("allow_plaintext_index", False))


def hnsw_ef_search():
    return int(_raw_biometric_cfg().get("hnsw_ef_search", 200))

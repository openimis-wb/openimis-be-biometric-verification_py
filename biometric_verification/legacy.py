"""
Availability of the legacy insuree/claim apps in the current assembly.

BiometricEmbedding (FK to insuree.Insuree) and ClaimFacialAudit (FK to
claim.Claim) are health-assembly-only models. Some assemblies (e.g. social
protection) never install those apps. Reading settings.INSTALLED_APPS is
safe at import time; django.apps.apps.is_installed() is not — it requires
the app registry to be populated, which is not yet true while models.py
of an installed app is itself being imported.
"""

from django.conf import settings

HAS_INSUREE = "insuree" in settings.INSTALLED_APPS
HAS_CLAIM = "claim" in settings.INSTALLED_APPS

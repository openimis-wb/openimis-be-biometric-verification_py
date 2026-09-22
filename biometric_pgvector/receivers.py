"""
Keeps BiometricVectorIndex in sync with BiometricTemplate: an active
embedding-kind template gets its side row upserted from the decrypted
vector; a superseded, deleted, or non-embedding template loses it
(docs/wb-biometric-dedup-seam.md §6.2).

Connected directly to django.db.models.signals in
BiometricPgvectorConfig.ready() — these are plain model signals, not the
core service-signal mechanism used for the deduplication seam.
"""

import logging

from django.db.models.signals import post_delete, post_save

logger = logging.getLogger(__name__)


def sync_one(instance):
    """Upsert or drop the side row for one BiometricTemplate instance."""
    from biometric import crypto
    from biometric.apps import BiometricConfig

    from .models import BiometricVectorIndex

    eligible = instance.kind == "embedding" and instance.validity_to is None and instance.vector is not None
    if not eligible:
        BiometricVectorIndex.objects.filter(template_id=instance.id).delete()
        return

    key = BiometricConfig.template_key if instance.encrypted else None
    vector = crypto.decrypt_vector(instance.vector, key)
    if not vector:
        BiometricVectorIndex.objects.filter(template_id=instance.id).delete()
        return

    BiometricVectorIndex.objects.update_or_create(
        template_id=instance.id,
        defaults={
            "modality": instance.modality,
            "provider": instance.provider,
            "model_name": instance.model_name,
            "dim": len(vector),
            "embedding": vector,
        },
    )


def _on_template_saved(sender, instance, **kwargs):
    # The index is a derived, best-effort structure: a sync failure must
    # never fail the enrol()/consolidate() write it is reacting to.
    try:
        sync_one(instance)
    except Exception:
        logger.exception("biometric_pgvector: failed to sync vector index for template %s", instance.id)


def _on_template_deleted(sender, instance, **kwargs):
    from .models import BiometricVectorIndex

    BiometricVectorIndex.objects.filter(template_id=instance.id).delete()


def connect():
    from biometric.models import BiometricTemplate

    post_save.connect(_on_template_saved, sender=BiometricTemplate, dispatch_uid="biometric_pgvector_sync_on_save")
    post_delete.connect(
        _on_template_deleted, sender=BiometricTemplate, dispatch_uid="biometric_pgvector_sync_on_delete"
    )

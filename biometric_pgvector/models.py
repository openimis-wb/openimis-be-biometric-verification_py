from django.db import models
from pgvector.django import VectorField

from biometric.models import BiometricTemplate


class BiometricVectorIndex(models.Model):
    """
    Plaintext copy of one active embedding template's vector, kept for ANN
    search (docs/wb-biometric-dedup-seam.md §6.2). Synced from BiometricTemplate
    by receivers.py — never written to directly. modality/provider/model_name
    are denormalised from the template so the gallery filter never needs to
    touch it; dim is per row since embedding carries no fixed dimension.
    """

    template = models.OneToOneField(
        BiometricTemplate,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="vector_index",
    )
    modality = models.CharField(max_length=16)
    provider = models.CharField(max_length=64)
    model_name = models.CharField(max_length=64)
    dim = models.IntegerField()
    embedding = VectorField()

    class Meta:
        db_table = "biometric_vector_index"
        indexes = [
            models.Index(fields=["modality", "provider", "model_name"], name="bvi_gallery_idx"),
        ]

    def __str__(self):
        return f"BiometricVectorIndex(template={self.template_id}, model_name={self.model_name})"

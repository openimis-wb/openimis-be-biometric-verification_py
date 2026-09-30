"""
This module's CandidateSource (docs/wb-biometric-dedup-seam.md §2.1, §3.5).

deduplication.sources is imported only inside try/except ImportError — the
two modules must each install standalone. When it isn't installed, this
file falls back to a local mirror of the same Candidate/Watermark/
CandidateSource shapes so BiometricCandidateSource and its tests still run;
apps.py never registers it anywhere in that case (there is no registry).
"""

import logging

logger = logging.getLogger(__name__)

try:
    from deduplication.sources import Candidate, CandidateSource, Watermark, order_pair
except ImportError:
    import abc
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Iterable, Optional

    @dataclass(frozen=True)
    class Watermark:
        updated_at: Optional[datetime] = None
        last_id: Optional[str] = None

    @dataclass(frozen=True)
    class Candidate:
        subject_model: str
        subject_a: str
        subject_b: str
        kind: str
        score: Optional[float]
        evidence: dict

    class CandidateSource(abc.ABC):
        kind: str

        @abc.abstractmethod
        def scan(self, since: Optional[Watermark]) -> Iterable[Candidate]:
            raise NotImplementedError

        @abc.abstractmethod
        def watermark(self) -> Watermark:
            raise NotImplementedError

    def order_pair(a: str, b: str):
        return (a, b) if a < b else (b, a)


class BiometricCandidateSource(CandidateSource):
    """
    Scans active templates for one modality, ranks each against the rest of
    the gallery, and yields a Candidate per match at or above
    DEDUP_THRESHOLD[modality]. watermark() advances on (date_updated, id).
    On the numpy and template paths the gallery is read and decrypted once per
    scan (services.Gallery); on the pgvector path each probe runs identify().
    Templates recorded under another preprocessing than the modality's
    provider (§6.13) are neither scanned nor matched.
    """

    kind = "biometric"

    def __init__(self, modality: str = "face"):
        self.modality = modality

    def scan(self, since: "Optional[Watermark]" = None):
        from . import crypto
        from .apps import BiometricConfig
        from .models import BiometricTemplate
        from .registry import ProviderRegistry
        from .services import Gallery, comparable_preprocessing, identify, log_preprocessing_skips

        threshold = BiometricConfig.dedup_threshold.get(self.modality, 0.0)
        provider = None
        gallery = None

        queryset = BiometricTemplate.objects.filter(
            modality=self.modality, validity_to__isnull=True,
        ).order_by("date_updated", "id")

        if since is not None and since.updated_at is not None:
            queryset = queryset.filter(date_updated__gte=since.updated_at)
            if since.last_id is not None:
                queryset = queryset.exclude(date_updated=since.updated_at, id__lte=since.last_id)

        seen_pairs = set()
        skipped = 0
        for template in queryset.iterator():
            if provider is None:
                provider = ProviderRegistry.get_provider(self.modality)
                # The pgvector path ranks in the database and decrypts nothing;
                # the numpy and template paths share one decrypted gallery.
                if not (provider.kind == "embedding" and BiometricConfig.vector_index == "pgvector"):
                    gallery = Gallery(provider, self.modality)
            # The gallery is filtered on preprocessing already; a probe row under
            # another preprocessing is left out here.
            if not comparable_preprocessing(template, provider):
                skipped += 1
                continue
            probe_vector, probe_template = self._probe(template, gallery, crypto, BiometricConfig.template_key)

            if gallery is not None:
                matches = gallery.rank(
                    vector=probe_vector, template=probe_template, exclude_subject=template.subject_id,
                )
            else:
                matches = identify(
                    self.modality,
                    vector=probe_vector,
                    template=probe_template,
                    scope=None,
                    exclude_subject=template.subject_id,
                )
            for match in matches:
                if match.score is None or match.score < threshold:
                    continue
                subject_a, subject_b = order_pair(template.subject_id, match.subject_id)
                pair_key = (subject_a, subject_b)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                yield Candidate(
                    subject_model=template.subject_model,
                    subject_a=subject_a,
                    subject_b=subject_b,
                    kind=self.kind,
                    score=match.score,
                    evidence={
                        "modality": self.modality,
                        "provider": template.provider,
                        "model_name": template.model_name,
                        "template_a": str(template.id),
                        "template_b": match.template_id,
                    },
                )
        log_preprocessing_skips(logger, skipped, self.modality, "BiometricCandidateSource.scan()")

    @staticmethod
    def _probe(template, gallery, crypto, key):
        """(vector, template) of a probe row: the gallery's decrypted value when the row is in it."""
        if gallery is not None:
            try:
                stored = gallery.stored_value(template.id)
            except KeyError:
                pass
            else:
                return (stored, None) if gallery.kind == "embedding" else (None, stored)
        row_key = crypto.row_key(template.encrypted, key)
        return crypto.decrypt_vector(template.vector, row_key), crypto.decrypt_bytes(template.template, row_key)

    def watermark(self):
        from .models import BiometricTemplate

        last = (
            BiometricTemplate.objects.filter(modality=self.modality, validity_to__isnull=True)
            .order_by("-date_updated", "-id")
            .values("date_updated", "id")
            .first()
        )
        if not last:
            return Watermark()
        return Watermark(updated_at=last["date_updated"], last_id=str(last["id"]))

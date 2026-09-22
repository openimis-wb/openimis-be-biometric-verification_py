"""
This module's CandidateSource (docs/wb-biometric-dedup-seam.md §2.1, §3.5).

deduplication.sources is imported only inside try/except ImportError — the
two modules must each install standalone. When it isn't installed, this
file falls back to a local mirror of the same Candidate/Watermark/
CandidateSource shapes so BiometricCandidateSource and its tests still run;
apps.py never registers it anywhere in that case (there is no registry).
"""

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
    Scans active templates for one modality, runs identify() for each against
    the rest of the gallery, and yields a Candidate per match at or above
    DEDUP_THRESHOLD[modality]. watermark() advances on (date_updated, id).
    """

    kind = "biometric"

    def __init__(self, modality: str = "face"):
        self.modality = modality

    def scan(self, since: "Optional[Watermark]" = None):
        from .apps import BiometricVerificationConfig
        from .models import BiometricTemplate
        from .services import identify

        threshold = BiometricVerificationConfig.dedup_threshold.get(self.modality, 0.0)

        queryset = BiometricTemplate.objects.filter(
            modality=self.modality, validity_to__isnull=True,
        ).order_by("date_updated", "id")

        if since is not None and since.updated_at is not None:
            queryset = queryset.filter(date_updated__gte=since.updated_at)
            if since.last_id is not None:
                queryset = queryset.exclude(date_updated=since.updated_at, id__lte=since.last_id)

        seen_pairs = set()
        for template in queryset.iterator():
            probe_vector = template.vector
            probe_template = template.template
            if template.encrypted:
                from . import crypto
                key = BiometricVerificationConfig.template_key
                probe_vector = crypto.decrypt_vector(probe_vector, key)
                probe_template = crypto.decrypt_bytes(probe_template, key)

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

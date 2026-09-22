from typing import Optional

from .base import Extracted, MatcherProvider


class DeviceReportedMatcher(MatcherProvider):
    """
    Registered provider for a modality matched on the device (e.g. fingerprint
    on a tablet SDK). extract() stores the device-supplied template as given;
    match() has no server-side implementation — verification for this modality
    goes through the device-reported path in services.verify() instead.
    """

    provider_name = "device_reported"
    default_threshold = 0.0

    def __init__(self, modality: str, default_threshold: float = 0.0, **kwargs):
        self.modality = modality
        self.default_threshold = default_threshold

    def extract(self, sample: bytes, position: Optional[str] = None) -> Extracted:
        return Extracted(template=sample)

    def match(self, probe: bytes, reference: bytes) -> float:
        raise NotImplementedError(
            "device_reported matching happens on the device; "
            "use the device-reported verify path (services.verify(device_score=...))."
        )

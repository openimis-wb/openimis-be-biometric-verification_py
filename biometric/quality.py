"""
Enrolment quality gate (docs/wb-biometric-dedup-seam.md §6.7).

assess() turns one sample and what the provider extracted from it into a
QualityVerdict: a list of measures, each recorded beside the limit it is judged
against, and a status derived from them. A measure whose value or limit is None
is recorded and not judged.

Face samples are read with NumPy, and with Pillow when it is importable:
- sharpness: variance of the 5-point Laplacian of the whole grayscale sample;
  judged against min_sharpness (100.0 by default).
- yaw / pitch / roll: provider-supplied angles in degrees, or roll from the
  eye line; yaw and roll are judged (20.0 by default), pitch is not.
- yaw_ratio: nose offset along the eye axis from 5-point landmarks; not judged
  by default.
- lower_face_uniformity: CIE L*a*b* spread inside the lower part of the
  provider's face box; not judged by default.
- provider_quality: Extracted.quality; judged only when min_quality is set.
Every other modality carries provider_quality only.

Nothing here imports a Django model; NumPy is imported inside the functions and
Pillow through _pil(), so the module imports without either.
"""

import io
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from django.core.exceptions import ImproperlyConfigured

logger = logging.getLogger(__name__)

ACCEPTED = "ACCEPTED"
REFUSED = "REFUSED"
NOT_ASSESSED = "NOT_ASSESSED"
VERDICT_VERSION = 1

MODES = ("advisory", "enforce")

BUILTIN_THRESHOLDS = {
    "face": {
        "min_sharpness": 100.0,
        "max_yaw": 20.0,
        "max_pitch": None,
        "max_roll": 20.0,
        "max_yaw_ratio": None,
        "min_lower_face_uniformity": None,
        "min_quality": None,
    },
}
DEFAULT_THRESHOLDS = {"min_quality": None}

# Lower-face region, as fractions of the face box: rows from the nose landmark
# (or LOWER_FACE_TOP of the box height when there is none) to the bottom of the
# box, columns inside LOWER_FACE_SIDE of either edge. With both mouth corners
# given, a band of MOUTH_HALF_HEIGHT + LIP_MARGIN box heights around the mouth
# line, widened by LIP_MARGIN beyond each corner, is cut out. A region under
# MIN_REGION_PIXELS pixels yields no value. The measure is recorded and not
# judged unless min_lower_face_uniformity is configured.
MIN_REGION_PIXELS = 50
LIP_MARGIN = 0.04
LOWER_FACE_TOP = 0.6
LOWER_FACE_SIDE = 0.2
MOUTH_HALF_HEIGHT = 0.08

SAMPLE_UNDECODABLE = "sample_undecodable"


@dataclass(frozen=True)
class QualityMeasure:
    """One measure and the limit it is judged against."""
    name: str
    value: Optional[float]
    limit: Optional[float]
    kind: str                  # "min": value >= limit passes; "max": abs(value) <= limit passes
    source: str                # "image" | "provider_pose" | "landmarks" | "provider"
    detail: str = ""

    @property
    def passed(self) -> Optional[bool]:
        """None when the measure is not judged; the bound itself passes."""
        if self.value is None or self.limit is None:
            return None
        if self.kind == "min":
            return self.value >= self.limit
        return abs(self.value) <= self.limit

    @property
    def reason(self) -> str:
        return f"{self.name}_below_min" if self.kind == "min" else f"{self.name}_above_max"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "limit": self.limit,
            "kind": self.kind,
            "passed": self.passed,
            "source": self.source,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class QualityVerdict:
    """What the gate concluded about one sample."""
    status: str
    mode: str
    modality: str
    reasons: List[str] = field(default_factory=list)
    measures: List[QualityMeasure] = field(default_factory=list)
    version: int = VERDICT_VERSION

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "mode": self.mode,
            "modality": self.modality,
            "reasons": list(self.reasons),
            "measures": [m.as_dict() for m in self.measures],
            "version": self.version,
        }


class QualityRefusedError(ValueError):
    """Raised by enrol() in enforce mode when the verdict is REFUSED; nothing is written."""

    def __init__(self, verdict: QualityVerdict):
        self.verdict = verdict
        # graphql-core copies .extensions onto the GraphQL error it reports.
        self.extensions = {"code": "BIOMETRIC_QUALITY_REFUSED", "verdict": verdict.as_dict()}
        super().__init__("Biometric sample refused by the quality gate: " + ", ".join(verdict.reasons))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _quality_cfg() -> dict:
    from .apps import BiometricConfig

    cfg = BiometricConfig.quality
    return cfg if isinstance(cfg, dict) else {}


def mode() -> str:
    """The configured gate mode; ImproperlyConfigured for any value outside MODES."""
    value = _quality_cfg().get("mode", "advisory")
    if value not in MODES:
        raise ImproperlyConfigured(
            f"BIOMETRIC['QUALITY']['mode'] must be one of {MODES}, got {value!r}."
        )
    return value


def thresholds_for(modality: str) -> dict:
    """Built-in thresholds for the modality, overridden key by key by the configured ones."""
    configured = _quality_cfg().get("modalities") or {}
    overrides = configured.get(modality) or {}
    return {**BUILTIN_THRESHOLDS.get(modality, DEFAULT_THRESHOLDS), **overrides}


def _limit(thresholds: dict, key: str) -> Optional[float]:
    value = thresholds.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ImproperlyConfigured(
            f"BIOMETRIC['QUALITY'] threshold {key!r} must be a number or None, got {value!r}."
        )


def _pil():
    """The PIL.Image module, or None when Pillow is not importable."""
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def pillow_available() -> bool:
    return _pil() is not None


def _warn_if_pillow_missing_in_enforce() -> None:
    """Logs a warning when enforce mode is configured but face images cannot be read."""
    if _quality_cfg().get("mode") == "enforce" and not pillow_available():
        logger.warning(
            "BIOMETRIC['QUALITY'] is in enforce mode but Pillow is not importable: "
            "face image measures are unavailable and never refuse a sample."
        )


# ---------------------------------------------------------------------------
# Image measures
# ---------------------------------------------------------------------------

def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def decode_image(sample: bytes):
    """
    (gray float32 HxW, rgb uint8 HxWx3, detail) for an image sample, as Pillow
    decodes it without EXIF transpose. On failure both arrays are None and
    detail is 'pillow_unavailable' or 'undecodable'.
    """
    Image = _pil()
    if Image is None:
        return None, None, "pillow_unavailable"
    import numpy as np

    try:
        with Image.open(io.BytesIO(sample)) as img:
            img.load()
            gray = np.asarray(img.convert("L"), dtype=np.float32)
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None, None, "undecodable"
    return gray, rgb, ""


def sharpness(gray) -> float:
    """Variance of the 5-point Laplacian over the reflect-padded image; 0.0 below 3x3."""
    import numpy as np

    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2 or gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    p = np.pad(gray, 1, mode="reflect")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * p[1:-1, 1:-1]
    return float(lap.astype(np.float64).var())


def srgb_to_lab(rgb):
    """sRGB (D65) uint8 or float 0-255 HxWx3 to CIE L*a*b*, float64."""
    import numpy as np

    c = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3.0 * delta ** 2) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def _point(face, name) -> Optional[Tuple[float, float]]:
    """A named landmark as a finite (x, y), else None."""
    raw = (face.landmarks or {}).get(name) if face is not None else None
    if raw is None:
        return None
    try:
        x, y = float(raw[0]), float(raw[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def lower_face_uniformity(rgb, face) -> Tuple[Optional[float], str]:
    """
    Median over the lower-face region of the summed absolute L*a*b* deviation
    from the region's median colour. High on a bare lower face (shading, beard),
    low on a uniformly coloured cover. (None, detail) when there is no box or
    the region is too small.
    """
    import numpy as np

    if face is None or face.box is None:
        return None, "no_face_box"
    try:
        x, y, w, h = (float(v) for v in face.box)
    except (TypeError, ValueError):
        return None, "geometry_invalid"
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None, "geometry_invalid"

    height, width = rgb.shape[:2]
    nose = _point(face, "nose")
    top = nose[1] if nose is not None else y + LOWER_FACE_TOP * h
    r0 = max(0, int(round(top)))
    r1 = min(height, int(round(y + h)))
    c0 = max(0, int(round(x + LOWER_FACE_SIDE * w)))
    c1 = min(width, int(round(x + (1.0 - LOWER_FACE_SIDE) * w)))
    if r1 <= r0 or c1 <= c0:
        return None, "region_too_small"

    region = np.zeros((height, width), dtype=bool)
    region[r0:r1, c0:c1] = True

    left, right = _point(face, "mouth_left"), _point(face, "mouth_right")
    if left is not None and right is not None:
        mouth_y = (left[1] + right[1]) / 2.0
        half = (MOUTH_HALF_HEIGHT + LIP_MARGIN) * h
        margin = LIP_MARGIN * h
        mr0 = max(0, int(math.floor(mouth_y - half)))
        mr1 = min(height, int(math.ceil(mouth_y + half)))
        mc0 = max(0, int(math.floor(min(left[0], right[0]) - margin)))
        mc1 = min(width, int(math.ceil(max(left[0], right[0]) + margin)))
        if mr1 > mr0 and mc1 > mc0:
            region[mr0:mr1, mc0:mc1] = False

    if int(region.sum()) < MIN_REGION_PIXELS:
        return None, "region_too_small"
    pixels = srgb_to_lab(rgb[region])
    deviation = np.abs(pixels - np.median(pixels, axis=0)).sum(axis=1)
    return float(np.median(deviation)), ""


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

def _eye_axis(face):
    """(roll degrees, midpoint, unit axis, inter-ocular distance) from the eyes, or a detail string."""
    a, b = _point(face, "left_eye"), _point(face, "right_eye")
    raw_a = (face.landmarks or {}).get("left_eye")
    raw_b = (face.landmarks or {}).get("right_eye")
    if raw_a is None or raw_b is None:
        return None
    if a is None or b is None:
        return "geometry_invalid"
    if a[0] > b[0]:
        a, b = b, a
    dx, dy = b[0] - a[0], b[1] - a[1]
    distance = math.hypot(dx, dy)
    if distance == 0.0:
        return "geometry_invalid"
    roll = math.degrees(math.atan2(dy, dx))
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return roll, mid, (dx / distance, dy / distance), distance


def pose_measures(face, thresholds: dict) -> List[QualityMeasure]:
    """
    yaw, pitch, roll and yaw_ratio. An angle the provider supplies wins
    (source 'provider_pose'); otherwise roll comes from the eye line and
    yaw_ratio from the nose offset along it (source 'landmarks'). An angle
    with no source is recorded with value None and detail 'no_pose'.
    """
    limits = {axis: _limit(thresholds, f"max_{axis}") for axis in ("yaw", "pitch", "roll")}
    pose = (face.pose or {}) if face is not None else {}
    axis_info = _eye_axis(face) if face is not None else None

    measures = []
    for axis in ("yaw", "pitch", "roll"):
        if axis in pose:
            value = _finite(pose[axis])
            measures.append(QualityMeasure(
                axis, value, limits[axis], "max", "provider_pose",
                "" if value is not None else "geometry_invalid",
            ))
        elif axis == "roll" and axis_info is not None:
            if isinstance(axis_info, str):
                measures.append(QualityMeasure(axis, None, limits[axis], "max", "landmarks", axis_info))
            else:
                measures.append(QualityMeasure(axis, axis_info[0], limits[axis], "max", "landmarks"))
        else:
            measures.append(QualityMeasure(axis, None, limits[axis], "max", "provider_pose", "no_pose"))

    if "yaw" not in pose and axis_info is not None and (face.landmarks or {}).get("nose") is not None:
        ratio_limit = _limit(thresholds, "max_yaw_ratio")
        nose = _point(face, "nose")
        if isinstance(axis_info, str) or nose is None:
            measures.append(QualityMeasure("yaw_ratio", None, ratio_limit, "max", "landmarks", "geometry_invalid"))
        else:
            _, mid, (ux, uy), distance = axis_info
            offset = (nose[0] - mid[0]) * ux + (nose[1] - mid[1]) * uy
            measures.append(QualityMeasure("yaw_ratio", offset / (distance / 2.0), ratio_limit, "max", "landmarks"))
    return measures


# ---------------------------------------------------------------------------
# Assessors: (sample, extracted, thresholds, server_extracted) -> (measures, forced reasons)
# ---------------------------------------------------------------------------

def _provider_quality(extracted, thresholds) -> QualityMeasure:
    raw = getattr(extracted, "quality", None)
    value = _finite(raw)
    detail = "not_reported" if raw is None else ("" if value is not None else "not_finite")
    return QualityMeasure("provider_quality", value, _limit(thresholds, "min_quality"), "min", "provider", detail)


def assess_face(sample, extracted, thresholds, *, server_extracted: bool):
    face = getattr(extracted, "face", None)
    forced = []
    gray, rgb, detail = decode_image(sample)
    if gray is None:
        if detail == "undecodable":
            if server_extracted:
                forced.append(SAMPLE_UNDECODABLE)
            else:
                detail = "sample_not_image"
        sharp = QualityMeasure("sharpness", None, _limit(thresholds, "min_sharpness"), "min", "image", detail)
        lower = QualityMeasure(
            "lower_face_uniformity", None, _limit(thresholds, "min_lower_face_uniformity"), "min", "image", detail,
        )
    else:
        sharp = QualityMeasure("sharpness", sharpness(gray), _limit(thresholds, "min_sharpness"), "min", "image")
        value, lower_detail = lower_face_uniformity(rgb, face)
        lower = QualityMeasure(
            "lower_face_uniformity", value, _limit(thresholds, "min_lower_face_uniformity"), "min", "image",
            lower_detail,
        )
    measures = [sharp, lower, *pose_measures(face, thresholds), _provider_quality(extracted, thresholds)]
    return measures, forced


def assess_reported(sample, extracted, thresholds, *, server_extracted: bool):
    return [_provider_quality(extracted, thresholds)], []


ASSESSORS = {"face": assess_face}


def assess(modality, sample, extracted, *, server_extracted=True, mode_value=None) -> QualityVerdict:
    """
    REFUSED when any judged measure fails (or a server-extracted sample does not
    decode), ACCEPTED when at least one measure is judged and none fails,
    NOT_ASSESSED otherwise.
    """
    if mode_value is None:
        mode_value = mode()
    elif mode_value not in MODES:
        raise ImproperlyConfigured(f"Quality gate mode must be one of {MODES}, got {mode_value!r}.")

    assessor = ASSESSORS.get(modality, assess_reported)
    measures, forced = assessor(sample, extracted, thresholds_for(modality), server_extracted=server_extracted)
    reasons = [m.reason for m in measures if m.passed is False] + list(forced)
    if reasons:
        status = REFUSED
    elif any(m.passed is not None for m in measures):
        status = ACCEPTED
    else:
        status = NOT_ASSESSED
    return QualityVerdict(status=status, mode=mode_value, modality=modality, reasons=reasons, measures=measures)

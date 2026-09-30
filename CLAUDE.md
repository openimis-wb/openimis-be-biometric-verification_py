# CLAUDE.md — openimis-be-biometric_verification

> **Maintenance rule**: Keep this file up to date as the codebase evolves.
> Every time a file is created, completed, or its design changes, update the
> **Implementation Status** table and the **Module Structure** tree in this
> document before finishing the task. Treat CLAUDE.md as the living spec —
> if it disagrees with the code, the code wins and this file must be corrected.

## Project Summary

This module adds **biometric identity verification** to openIMIS using face recognition. The primary use case is verifying an insuree's identity at point of service by comparing a live webcam frame against the reference photo stored at enrollment.

The architecture uses a **provider pattern**: DeepFace is the default local provider, but any external licensed API (AWS Rekognition, Azure Face API, Neurotechnology, etc.) can be plugged in via configuration without touching the module code.

---

## Key Decisions

- **Module name**: `openimis-be-biometric_verification` (not `face_*` — extensible to fingerprint and other modalities later)
- **Default local provider**: DeepFace with ArcFace model
  - Actively maintained (2024 research publication)
  - Wraps multiple best-in-class models: ArcFace, Facenet512, GhostFaceNet, Buffalo_L (InsightFace's model), SFace
  - Pure pip install, no separate service needed
  - **InsightFace pip package was rejected** — last PyPI release December 2022, stale for production use
- **Video stream strategy**: Single frame capture + GraphQL mutation (no WebSocket needed for openIMIS point-of-service use case)
- **Embedding storage**: Pre-compute and store face embeddings at enrollment in a `JSONField` on a dedicated `BiometricEmbedding` table — verification at service point becomes a vector comparison, not a full model inference
- **No FastAPI**: Uses Django views + GraphQL mutations (existing openIMIS stack)
- **No hand-written migrations**: all migrations are generated via `manage.py makemigrations`

---

## Implementation Status

| File | Status | Notes |
|---|---|---|
| `apps.py` | Done | Two-tier config: `ModuleConfiguration` DB + `settings.BIOMETRIC_VERIFICATION` override |
| `schema.py` | Done | `verifyFace` and `computeInsureeEmbedding` mutations with output types |
| `models.py` | Done | `BiometricEmbedding` model |
| `migrations/` | Done | Generated via `makemigrations`; copy from site-packages after generation |
| `services.py` | Done | `BiometricService` facade — verify_face, compute_insuree_embedding, photo fetch |
| `registry.py` | Done | `ProviderRegistry` with singleton instances; built-ins registered at import time |
| `providers/__init__.py` | Done | Package marker |
| `providers/base.py` | Done | `BaseBiometricProvider` ABC, `VerificationResult` dataclass, `_cosine_distance` |
| `providers/deepface_provider.py` | Done | DeepFace implementation (ArcFace default) |
| `providers/external/` | Pending | AWS, Azure, custom provider templates |
| `admin.py` | Pending | Django admin for `BiometricEmbedding` and `ModuleConfiguration` |
| `views.py` | Done | Stub — kiosk UI moved to frontend module |
| `urls.py` | Done | Empty — kiosk UI served by openimis-fe-biometric-verification |
| `SECURITY.md` | Done | Risk assessment and mandatory mitigations for the public endpoint |
| `tests/__init__.py` | Done | Package marker |
| `tests/test_base_provider.py` | Done | VerificationResult, _cosine_distance, verify_from_embedding |
| `tests/test_registry.py` | Done | register, get_active_provider, singleton, KeyError/TypeError |
| `tests/test_services.py` | Done | _decode_frame, _fetch_insuree_photo, verify_face, compute_insuree_embedding |
| `tests/test_schema.py` | Done | GraphQL mutation permission guards + service delegation |

---

## Module Structure

```
openimis-be-biometric_verification_py/
├── biometric_verification/
│   ├── __init__.py            ✅
│   ├── apps.py                ✅ AppConfig + two-tier settings loading
│   ├── schema.py              ✅ GraphQL mutations: verifyFace, computeInsureeEmbedding
│   ├── models.py              ✅ BiometricEmbedding model
│   ├── services.py            ✅ BiometricService facade (verify_face, compute_embedding)
│   ├── registry.py            ✅ ProviderRegistry — singleton instances, auto-registers built-ins
│   ├── providers/
│   │   ├── __init__.py        ✅
│   │   ├── base.py            ✅ BaseBiometricProvider ABC + VerificationResult dataclass
│   │   ├── deepface_provider.py ✅ DeepFace implementation (ArcFace default)
│   │   └── external/
│   │       ├── aws_provider.py       🔲
│   │       ├── azure_provider.py     🔲
│   │       └── custom_provider.py   🔲 Template for licensed 3rd-party SDKs
│   ├── admin.py               🔲 Django admin registration
│   ├── views.py               ✅ Public kiosk page (no auth)
│   ├── migrations/            ✅ Generated via makemigrations
│   ├── tests/
│   │   ├── __init__.py        ✅
│   │   ├── test_base_provider.py ✅ VerificationResult, _cosine_distance, verify_from_embedding
│   │   ├── test_registry.py   ✅ ProviderRegistry — register, singleton, errors
│   │   ├── test_services.py   ✅ BiometricService — decode, fetch photo, verify, embed
│   │   └── test_schema.py     ✅ GraphQL mutations — permissions + service delegation
│   └── urls.py                ✅ /biometric_verification/verify/ kiosk route
├── setup.py                   ✅ extras_require: deepface / aws / azure
├── README.md                  ✅
└── CLAUDE.md                  ✅ This file
```

---

## Configuration

### Priority order (highest wins)

1. `django.conf.settings.BIOMETRIC_VERIFICATION` — set in `settings.py`, applied at process startup
2. `ModuleConfiguration` database record — editable at runtime via Django admin
3. `DEFAULT_CFG` in `apps.py` — compile-time fallback

### Example `settings.py` block

```python
BIOMETRIC_VERIFICATION = {
    "PROVIDER": "deepface",
    "PROVIDER_CONFIG": {
        "model_name": "ArcFace",          # swap model here, no code change
        "detector_backend": "opencv",     # opencv | retinaface | mtcnn
    },
    "STORE_EMBEDDINGS": True,             # pre-compute at enrollment
    "SIMILARITY_THRESHOLD": 0.68,        # tune per deployment context
    "MAX_IMAGE_SIZE_PX": 1024,
}
```

Accepts both `UPPER_CASE` and `lower_case` keys.

---

## Provider Interface Contract

Every provider must implement `base.BaseBiometricProvider`:

```python
class BaseBiometricProvider(ABC):

    provider_name: str

    @abstractmethod
    def verify(probe_image, reference_image, threshold=None) -> VerificationResult:
        # Returns: verified (bool), confidence (0-100), distance (float), provider (str)
        pass

    @abstractmethod
    def get_embedding(image) -> list[float]:
        # Returns: vector (list[float]), model (str), provider (str)
        pass

    def verify_from_embedding(probe_image, reference_embedding) -> VerificationResult:
        # Default: recomputes probe embedding and does cosine distance
        # Override if the provider has a native optimized path
        pass

    def health_check() -> bool:
        pass
```

`VerificationResult` fields: `verified`, `confidence`, `distance`, `provider`, `metadata`, `error`

---

## DeepFace Provider Configuration

DeepFace wraps multiple models — configure via `PROVIDER_CONFIG`:

| Model | Accuracy | Speed (CPU) | Notes |
|---|---|---|---|
| `ArcFace` | ⭐⭐⭐⭐⭐ | Medium | **Recommended default** |
| `Facenet512` | ⭐⭐⭐⭐ | Medium | Good balance |
| `Buffalo_L` | ⭐⭐⭐⭐⭐ | Medium | InsightFace model via DeepFace |
| `GhostFaceNet` | ⭐⭐⭐⭐ | Fast | Best for CPU-constrained deployments |
| `SFace` | ⭐⭐⭐ | Very fast | Minimum viable, edge devices |

**Important**: DeepFace loads the model into memory on first call (~3-5s). This happens once at worker startup — not on every request.

---

## GraphQL API

### Mutation: `verifyFace`
Compares a base64 webcam frame against the stored reference for an insuree.

```graphql
mutation {
  verifyFace(insureeUuid: "...", frameB64: "data:image/jpeg;base64,...") {
    verified
    confidence
    distance
    provider
    error
  }
}
```

### Mutation: `computeInsureeEmbedding`
Pre-computes and stores the embedding for an insuree photo (call at enrollment).

```graphql
mutation {
  computeInsureeEmbedding(insureeUuid: "...") {
    success
    model
    provider
    error
  }
}
```

---

## Data Model

```python
class BiometricEmbedding(models.Model):
    id           = AutoField (PK)
    uuid         = UUIDField (unique, indexed)
    insuree      = OneToOneField("insuree.Insuree", CASCADE)
    embedding    = JSONField()          # float vector, dimension depends on model
    model_name   = CharField(64)        # e.g. "ArcFace"
    provider     = CharField(64)        # e.g. "deepface"
    computed_at  = DateTimeField(auto_now=True)
    validity_from= DateTimeField(auto_now_add=True)
    validity_to  = DateTimeField(null, blank, indexed)  # NULL = active
```

- `db_table = "biometric_embedding"` (new table, not a legacy SQL Server table)
- `validity_to = NULL` means the embedding is currently active
- On re-enrollment or photo update: set `validity_to` on the old row, insert a new one

---

## Dependencies

Managed via `setup.py` `extras_require` — **not** in the main `requirements.txt`:

```bash
# Local DeepFace inference
pip install "openimis-be-biometric_verification[deepface]"

# AWS Rekognition
pip install "openimis-be-biometric_verification[aws]"

# Azure Face API
pip install "openimis-be-biometric_verification[azure]"
```

| Extra | Packages |
|---|---|
| `deepface` | `deepface>=0.0.93`, `opencv-python-headless>=4.9.0`, `tf-keras`, `numpy` |
| `aws` | `boto3` |
| `azure` | `azure-cognitiveservices-vision-face`, `msrest` |

---

## Adding a New Provider (External / Licensed)

1. Create `providers/external/my_provider.py`
2. Extend `BaseBiometricProvider`, implement `verify()` and `get_embedding()`
3. Register in `registry.py`:
   ```python
   ProviderRegistry.register("my_provider", MyProvider)
   ```
4. Update settings:
   ```python
   BIOMETRIC_VERIFICATION = {
       "PROVIDER": "my_provider",
       "PROVIDER_CONFIG": { ... }
   }
   ```

No other code changes required.

---

## Performance Notes

- **Enrollment**: `computeInsureeEmbedding` runs once per insuree, stores the vector. ~1-3s per photo.
- **Verification**: With stored embeddings, only the probe frame goes through model inference. ~200-500ms on CPU.
- **Model warmup**: Pre-load the DeepFace model in `AppConfig.ready()` to avoid cold-start latency on the first real request.
- **Deployment context**: For CPU-only servers (common in LMIC), prefer `GhostFaceNet` or `SFace` over `ArcFace` if latency is a concern. Tune `SIMILARITY_THRESHOLD` per country — lighting conditions and photo quality at enrollment vary significantly.

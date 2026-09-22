# openimis-be-biometric_verification

Biometric identity verification module for [openIMIS](https://openimis.org). Verifies an insuree's identity at point of service by comparing a live webcam capture against the reference photo stored at enrollment.

Built with a **provider pattern** — DeepFace is the default local engine, but any external licensed API can be swapped in via configuration.

---

## Features

- 1:1 face verification (probe vs. stored reference photo)
- Pre-computed embedding storage for fast point-of-service verification
- Pluggable provider architecture: local (DeepFace) or external (AWS, Azure, custom licensed SDK)
- GraphQL API — no new REST endpoints, integrates with existing openIMIS stack
- Configurable model and threshold per deployment context

---

## Quick Start

```bash
pip install openimis-be-biometric_verification
```

Add to `openimis.json`:
```json
{
  "modules": ["biometric_verification"]
}
```

Configure in `settings.py`:
```python
BIOMETRIC_VERIFICATION = {
    "PROVIDER": "deepface",
    "PROVIDER_CONFIG": {
        "model_name": "ArcFace",
        "detector_backend": "opencv"
    },
    "STORE_EMBEDDINGS": True,
    "SIMILARITY_THRESHOLD": 0.68,
}
```

Run migrations:
```bash
python manage.py migrate biometric_verification
```

`BiometricEmbedding` and `ClaimFacialAudit` (the legacy insuree 1:1 flow) are
only defined, and their migrations only create tables, when `insuree`/`claim`
are themselves installed. In an assembly without them (e.g. social
protection), those two models and their migrations are no-ops; the
multimodal models/services below are unaffected either way.

---

## Usage

**At enrollment** — pre-compute and store the insuree's face embedding:
```graphql
mutation {
  computeInsureeEmbedding(insureeUuid: "...") {
    success
    model
  }
}
```

**At point of service** — verify identity from a webcam frame:
```graphql
mutation {
  verifyFace(insureeUuid: "...", frameB64: "data:image/jpeg;base64,...") {
    verified
    confidence
    provider
  }
}
```

---

## Switching Providers

Change `PROVIDER` in `settings.py` — no code changes needed:

| Value | Description |
|---|---|
| `deepface` | Local inference via DeepFace (default) |
| `aws_rekognition` | AWS Rekognition cloud API |
| `azure_face` | Azure Face API |
| Custom | Any class implementing `BaseBiometricProvider` |

See `CLAUDE.md` for full provider implementation guide.

---

## Requirements

- Python 3.8+
- openIMIS backend (Django)
- `deepface`, `opencv-python-headless`, `tf-keras`

---

## Multimodal identity + deduplication (§3 of the biometric/dedup contract)

Alongside the legacy insuree 1:1 flow above, this module addresses any
subject as `(subject_model, subject_id)` — e.g. `"individual.Individual"` in
social protection, `"insuree.Insuree"` in health — and supports several
modalities (face, fingerprint, voice, iris, palmvein), each enrolled,
verified and identified independently.

### Configuration

`BIOMETRIC_VERIFICATION` (or the `biometric_verification` `ModuleConfiguration`
row) accepts, in addition to the legacy keys above:

```python
BIOMETRIC_VERIFICATION = {
    "SUBJECT_MODEL": "individual.Individual",
    "MODALITIES": {
        "face":        {"provider": "deepface",        "threshold": 0.68},
        "fingerprint": {"provider": "device_reported",  "threshold": 48},
    },
    "VECTOR_INDEX": "numpy",   # "numpy" (always available) | "pgvector" (only if the package is installed)
    "TEMPLATE_KEY": None,      # Fernet key (cryptography.fernet); templates/vectors encrypted at rest when set
    "REQUIRE_CONSENT": False,
    "DEDUP_THRESHOLD": {"face": 0.62},   # similarity at/above which a dedup candidate is emitted
    "FUSION": {
        "weights": {"face": 1.0},
        "thresholds": {"accept": 0.7, "review": 0.6},
        "floors": {},
        "floor_decision": "review",
    },
    "GQL_BIOMETRIC_ENROL_PERMS": ["174001"],
    "GQL_BIOMETRIC_VERIFY_PERMS": ["174002"],
    "GQL_BIOMETRIC_IDENTIFY_PERMS": ["174003"],
    "GQL_BIOMETRIC_READ_PERMS": ["174004"],
}
```

### Providers

- `providers.base.ModalityProvider` — common base (`modality`, `provider_name`,
  `kind`, `default_threshold`, `extract()`); `EmbeddingProvider` adds
  `distance()`/`similarity()`, `MatcherProvider` adds `match()`.
- `providers.deepface_provider.DeepFaceProvider` — also an `EmbeddingProvider`
  for `modality="face"` (registered under `("face", "deepface")`).
- `providers.device_reported.DeviceReportedMatcher` — stores a device-supplied
  template as given; matching happens on the device, not the server. Registered
  for every documented modality out of the box.
- `providers.fake.FakeEmbeddingProvider` / `FakeMatcherProvider` — deterministic,
  test-only providers (no ML model loaded).

Register a custom one with `ProviderRegistry.register_modality(modality, name,
provider_class)`; resolve the one configured for a modality with
`ProviderRegistry.get_provider(modality)`.

### Services (`biometric_verification.services`)

- `enrol(subject_model, subject_id, modality, sample, *, position=None, actor, metadata=None, device_template=None)`
  — extracts (or accepts a device template for) one sample, superseding any
  previous active row on the same key. Refused when `REQUIRE_CONSENT` and no
  `BiometricConsent` was granted.
- `verify(subject_model, subject_id, modality, *, sample=None, device_score=None, ..., actor)`
  — server path (extract + compare) or device path (`device_score` checked
  against the modality threshold). Always writes a `BiometricVerification` row.
- `identify(modality, *, sample=None, vector=None, template=None, top_k=5, scope=None, exclude_subject=None)`
  — ranks the gallery for one modality/provider/model. NumPy cosine path always
  available; `VECTOR_INDEX="pgvector"` routes to a lazy pgvector path instead.
- `fuse(scores, *, weights=None, thresholds=None, floors=None, floor_decision=None, required=frozenset())`
  — multimodal decision (`accept`/`review`/`reject`), tighten-only.
- `consolidate(subject_model, kept_id, retired_id, *, actor)` — re-points
  `retired`'s active templates to `kept` (or supersedes on key collision).
  Bound to the `deduplication.subject_merged` service signal.
- `templates_of(subject_model, subject_id, *, modality=None, actor, purpose="read")`
  — the only sanctioned plaintext read path; decrypts and logs the access.
- `purge(now=None, *, actor="retention")` — erases templates past
  `BiometricRetentionPolicy.template_retention_days`; no-op unless the policy
  has `purge_enabled` and a retention window set. Run via the `biometric_purge`
  management command.

### GraphQL

Mutations `enrolBiometric`, `verifyBiometric`, `recordBiometricConsent`;
queries `identifyBiometric(modality, sample, topK, excludeSubject)`,
`biometricTemplates(subjectModel, subjectId)` (metadata only — never
plaintext vectors/templates), `biometricVerifications(subjectModel, subjectId)`.
Samples travel base64 (optionally with a `data:...;base64,` prefix).

### Deduplication seam

This module registers a `BiometricCandidateSource` (kind `"biometric"`) with
`deduplication.sources` in `AppConfig.ready()`, guarded by
`try/except ImportError` — the deduplication package is optional. It scans
active templates for one modality, runs `identify()` against the rest of the
gallery, and yields a candidate per match at or above `DEDUP_THRESHOLD`.

---

## TODO / Known Limitations (POC)

### Claim Code Mutability
**Current implementation uses `claim_code` to link facial audits to claims during the verification flow.**

⚠️ **POC Limitation:** The claim code (`claim.code`) can potentially be modified during the claim processing workflow. If the code changes after facial audits are created, those audits will remain linked via the foreign key to the claim record (by ID), but the code used for lookup during WebSocket verification will no longer match.

**Production recommendation:** Switch to using `claim.uuid` instead of `claim.code` for linking facial audits, as UUIDs are immutable and more reliable for foreign key relationships. The current implementation prioritizes the QR code workflow where the claim code is more user-visible, but this comes at the cost of potential lookup failures if codes are reassigned.

**Affected files:**
- `consumers/biometric_consumer.py` — `_create_facial_audit()` method
- Frontend QR code dialog and verification page

---

## License

LGPL-3.0 — consistent with openIMIS module licensing.

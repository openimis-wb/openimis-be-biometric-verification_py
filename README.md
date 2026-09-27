# openimis-be-biometric_verification

This distribution ships **three independent Django apps**:

| App | Assembly | Subject | What it does |
|---|---|---|---|
| `biometric` | Any (social protection, generic) | `(subject_model, subject_id)`, e.g. `"individual.Individual"` | Multimodal identity: enrol/verify/identify across face, fingerprint, voice, iris, palmvein; deduplication candidate source |
| `biometric_pgvector` | Optional, on top of `biometric` | — (indexes `biometric.BiometricTemplate`) | Postgres `pgvector` ANN index for `identify()`, for galleries too large for the NumPy path |
| `biometric_verification` | Health only | `insuree.Insuree` / `claim.Claim` | Legacy 1:1 face verification at point of service, claim facial-audit trail |

None of them import each other at module import time. An assembly installs
the app(s) matching what it has: `biometric` needs nothing beyond
`openimis-be-core`; `biometric_pgvector` needs `biometric` installed and a
Postgres server with the `vector` extension available; `biometric_verification`
needs `insuree` and `claim` installed (its models FK to them unconditionally).
A health assembly with social-protection features can install any combination
of the three side by side.

---

## `biometric` — multimodal identity + deduplication

Built with a **provider pattern**: DeepFace is the default local face engine,
device-reported matching covers modalities matched on a tablet SDK
(fingerprint), and any other modality plugs in via `ModalityProvider`.

### Features

- Enrol/verify/identify across any modality, addressed by
  `(subject_model, subject_id)` — no FK, no contenttypes dependency
- Embedding (vector, cosine similarity) and template (opaque bytes, vendor
  matcher) providers, both on the same similarity scale (higher = more similar)
- Server-side verification (extract + compare) or device-reported verification
  (a tablet reports its own score, checked against the configured threshold)
- Multimodal fusion (`fuse()`) — weighted, tighten-only accept/review/reject
- Deduplication candidate source, registered with `deduplication.sources`
  when that module is installed
- At-rest encryption of vectors/templates (Fernet) when `TEMPLATE_KEY` is set
- Retention/purge, with an audit trail (`BiometricAccessLog`) and tombstones
  (`BiometricErasure`) on erasure
- Optional hash-chained audit events with alert rules (off by default)

### Quick start

```bash
pip install openimis-be-biometric_verification
```

Add to `openimis.json`:
```json
{
  "modules": ["biometric"]
}
```

Run migrations:
```bash
python manage.py migrate biometric
```

### Configuration

`BIOMETRIC` (or the `biometric` `ModuleConfiguration` row):

```python
BIOMETRIC = {
    "SUBJECT_MODEL": "individual.Individual",
    "MODALITIES": {
        "face":        {"provider": "deepface",        "threshold": 0.32},
        "fingerprint": {"provider": "device_reported",  "threshold": 48},
    },
    "VECTOR_INDEX": "numpy",   # "numpy" (always available) | "pgvector" (requires the biometric_pgvector app)
    "TEMPLATE_KEY": None,      # Fernet key (cryptography.fernet); templates/vectors encrypted at rest when set
    "REQUIRE_CONSENT": False,
    "DEDUP_THRESHOLD": {"face": 0.62},   # similarity at/above which a dedup candidate is emitted
    "FUSION": {
        "weights": {"face": 1.0},
        "thresholds": {"accept": 0.7, "review": 0.6},
        "floors": {},
        "floor_decision": "review",
    },
    "QUALITY": {               # enrolment quality gate; nested keys are lowercase
        "mode": "advisory",    # "advisory" (store the verdict) | "enforce" (refuse a REFUSED sample)
        "modalities": {
            "face": {"min_sharpness": 100.0, "max_yaw": 20.0, "max_pitch": None, "max_roll": 20.0,
                     "max_yaw_ratio": None, "min_lower_face_uniformity": None, "min_quality": None},
            # any modality: {"min_quality": None}  (provider-reported 0-100 quality)
        },
    },
    "RISK_PROFILES": {},       # named tighten-only overrides of FUSION / MODALITIES thresholds
    "IMPERSONATION_PROBE": {   # 1:N search inside server-path verify(); nested keys are lowercase
        "enabled": False,      # master switch; False leaves verify() unchanged
        "modalities": ["face"],  # modalities whose server-path verify() runs the probe
        "top_k": 5,            # foreign subjects kept
        "thresholds": {},      # {modality: score}; else DEDUP_THRESHOLD, then MODALITIES threshold, then provider default
        "margin": None,        # None flags any foreign match >= threshold; a float also requires score >= claimed - margin
    },
    "GQL_BIOMETRIC_ENROL_PERMS": ["174001"],
    "GQL_BIOMETRIC_VERIFY_PERMS": ["174002"],
    "GQL_BIOMETRIC_IDENTIFY_PERMS": ["174003"],
    "GQL_BIOMETRIC_READ_PERMS": ["174004"],
}
```

`face`'s default threshold (0.32) is on the similarity scale (1.0 = identical,
0.0 = orthogonal) — not the legacy `biometric_verification` distance scale.

### Enrolment quality gate

`enrol()` runs `biometric.quality.assess()` on every sample after extraction and
stores the verdict on `BiometricTemplate.quality_verdict` (`NULL` on rows enrolled
before the gate existed):

```
{"status": "ACCEPTED" | "REFUSED" | "NOT_ASSESSED", "mode": "advisory" | "enforce",
 "modality": "face", "reasons": ["sharpness_below_min"],
 "measures": [{"name", "value", "limit", "kind", "passed", "source", "detail"}], "version": 1}
```

- A measure is judged only when both its value and its limit are set; `passed` is
  `null` otherwise. `NOT_ASSESSED` means no measure was judged.
- Face: sharpness (Laplacian variance of the whole grayscale sample), head pose
  (provider angles, or roll from the eye line), a lower-face uniformity measure
  inside the provider's face box, and the provider-reported quality. Other
  modalities: the provider-reported quality only. Thresholds merge per key over
  the defaults above.
- `advisory` (default) stores the verdict and changes nothing else. `enforce`
  raises `QualityRefusedError` on a `REFUSED` verdict before anything is
  superseded or written; over GraphQL, `enrolBiometric` returns `null` with
  `errors[0].extensions = {"code": "BIOMETRIC_QUALITY_REFUSED", "verdict": {...}}`.
- Providers pass face geometry on `Extracted.face` (`FaceGeometry`: `box` as
  x, y, width, height; `landmarks` `left_eye`, `right_eye`, `nose`, `mouth_left`,
  `mouth_right`; `pose` yaw/pitch/roll in degrees), in pixels of the sample as
  Pillow decodes it without EXIF transpose. Geometry is never stored. The DeepFace
  provider supplies none, so its samples get sharpness and provider quality only.
- Pillow is optional. Without it, image measures are unavailable and never refuse;
  the app logs a warning at startup when `enforce` is configured.

### Risk profiles

`BIOMETRIC["RISK_PROFILES"]` names partial overrides that may only tighten the
base rules:

```python
"RISK_PROFILES": {
    "high_risk": {
        "thresholds": {"accept": 0.9},          # accept / review, may only rise
        "floors": {"fingerprint": 60},          # may only rise
        "floor_decision": "reject",             # "review" -> "reject" only
        "required": ["fingerprint"],            # added to the caller's set
        "modality_thresholds": {"face": 0.8},   # raw provider scale, configured modalities only
    },
},
```

- `fuse(scores, risk_profile="high_risk")` merges the profile key by key over
  the rules it resolved (max / stricter / union); `Decision.risk_profile`
  carries the name.
- `verify(..., risk_profile="high_risk")` and the `verifyBiometric(riskProfile:)`
  argument raise the modality threshold to the profile's value; the
  `BiometricVerification` row records the name and the effective threshold.
- `weights` cannot be overridden. A profile looser than the base is logged at
  startup and raises `RiskProfileError` when used; an unknown name raises
  `UnknownRiskProfileError` before anything is extracted or written, and is a
  GraphQL error rather than `verified: false`.

### Impersonation probe

With `BIOMETRIC["IMPERSONATION_PROBE"]["enabled"]` and the modality listed,
the server path of `verify()` ranks the extracted probe against the whole
gallery of the modality, drops the claimed `(subject_model, subject_id)` and
reports foreign subjects scoring at or above the probe threshold. The 1:1
`score`, `threshold` and `verified` are never changed by it.

- `VerificationResult.impersonation` is the `ImpersonationProbe` (`status`
  `ok` | `failed`, `suspected`, `best_match`, `candidates`, `claimed_score`),
  or `None` when the probe did not run. A probe that raises records `failed`
  and never fails `verify()`; its `error` is the exception class name and the
  fixed text `impersonation probe failed`, never the exception message.
- The `BiometricVerification` row stores `impersonation_status`,
  `impersonation_suspected`, `impersonation_subject_model`,
  `impersonation_subject_id`, `impersonation_score` and
  `impersonation_evidence`.
- On a suspicion, the service signal `biometric.impersonation_suspected`
  fires after commit. Bind it with `ServiceSignalBindType.AFTER` and read
  `kwargs["result"]`: `verification_id`, `subject_model`, `subject_id`,
  `modality`, `matched_subject_model`, `matched_subject_id`,
  `matched_template_id`, `matched_score`, `claimed_score`, `threshold`,
  `margin`, `actor`, `device_id`, `context`. The payload names two subjects;
  the subscriber applies its own access rights.
- GraphQL: `impersonation` on `verifyBiometric` and on `biometricVerifications`
  rows; the matched subject's identity and the candidate list need
  `GQL_BIOMETRIC_IDENTIFY_PERMS`.
- An OPEN duplicate pair is not suppressed: verifying either subject raises a
  suspicion naming the other until the pair is resolved.

### Audit chain and alerts

Off by default. With it on, enrolment, verification, identification by a
named user, template reads and listings, consolidation, purge and alert
triage each append a `BiometricAuditEvent` to a hash chain, and three alert
rules watch the chain.

```python
BIOMETRIC = {
    "AUDIT": {
        "enabled": True,
        "rules": {
            # Each kind overrides its defaults key by key; these are the defaults.
            "FAILED_VERIFICATIONS": {"threshold": 3, "window_minutes": 60, "per_modality": True},
            "IMPERSONATION_SUSPECTED": {"severity": "HIGH"},
            "ACCESS_BURST": {"max_events": 200, "window_minutes": 60,
                             "actions": ["template.read", "template.list", "identify"]},
        },
    },
}
```

- Each event stores `sequence`, `prev_hash` and
  `hash = sha256(prev_hash || canonical JSON of the row)`. Payloads carry
  identifiers, scores and counts. `sample`, `vector`, `template` and the
  other biometric keys are refused, as are binary values and numeric lists
  longer than 16.
- Appends serialize on a Postgres advisory lock that is held until the
  surrounding transaction commits. Each producer records its event as the
  last write of its transaction.
- Rules run after commit. A failing rule is logged and never fails the
  biometric operation.
  - `FAILED_VERIFICATIONS`: failed verifies of one subject within the window.
  - `IMPERSONATION_SUSPECTED`: every `impersonation.suspected` event, which
    `verify()` records when the impersonation probe suspects someone.
  - `ACCESS_BURST`: one account's reads within the window.

  A repeat bumps the open alert, including an acknowledged one. Resolving
  closes an alert, and the next occurrence opens a new one.
- An invalid `AUDIT` config is logged at startup. The rule it breaks is
  skipped, and its error is logged each time the rules run.
- Rights: `GQL_BIOMETRIC_AUDIT_PERMS` (`174005`) reads
  `biometricAuditEvents` and `biometricAlerts`. `GQL_BIOMETRIC_ALERT_PERMS`
  (`174006`) runs `acknowledgeBiometricAlert(id)` and
  `resolveBiometricAlert(id, note)`. No role holds them until a deployment
  grants them. Other subjects' identities in `identify` and impersonation
  payloads need `GQL_BIOMETRIC_IDENTIFY_PERMS` as well.

Verify the chain with `python manage.py biometric_audit_verify`. It prints
`biometric audit chain intact over N event(s); head sequence S hash H`, or
fails naming the first divergence. Store `S` and `H` somewhere this
database's writers cannot change, and pass them back on the next run. The
chain may have grown in between; the run fails only if event `S` is gone or
no longer holds `H`:

```bash
python manage.py biometric_audit_verify --expected-sequence S --expected-head H
```

The walk detects an altered, deleted or relinked row. It cannot detect
deletion of the newest rows; only a head recorded outside the database
reveals that, for the events up to that head. Record the newest head after
each run to cover the events appended since. It also cannot detect a rewrite by someone who holds both the
database and the application. There is no timestamp authority. Changing
`TIME_ZONE` after events exist changes every recomputed hash.

### Providers

- `providers.base.ModalityProvider` — common base (`modality`, `provider_name`,
  `kind`, `default_threshold`, `extract()`); `EmbeddingProvider` adds
  `distance()`/`similarity()`, `MatcherProvider` adds `match()`.
- `providers.deepface_provider.DeepFaceProvider` — `EmbeddingProvider` for
  `modality="face"` (registered under `("face", "deepface")`), similarity scale,
  default threshold 0.32.
- `providers.device_reported.DeviceReportedMatcher` — stores a device-supplied
  template as given; matching happens on the device, not the server. Registered
  for every documented modality out of the box.
- `providers.fake.FakeEmbeddingProvider` / `FakeMatcherProvider` — deterministic,
  test-only providers (no ML model loaded).

Register a custom one with `ProviderRegistry.register_modality(modality, name,
provider_class)`; resolve the one configured for a modality with
`ProviderRegistry.get_provider(modality)`.

### Services (`biometric.services`)

- `enrol(subject_model=None, subject_id, modality, sample, *, position=None, actor, metadata=None, device_template=None)`
  — extracts (or accepts a device template for) one sample, superseding any
  previous active row on the same key. Refused when `REQUIRE_CONSENT` and no
  `BiometricConsent` was granted. `subject_model` defaults to
  `BIOMETRIC["SUBJECT_MODEL"]`.
- `verify(subject_model=None, subject_id, modality, *, sample=None, device_score=None, ..., actor)`
  — server path (extract + compare) or device path (`device_score` checked
  against the modality threshold). Always writes a `BiometricVerification` row.
- `identify(modality, *, sample=None, vector=None, template=None, top_k=5, scope=None, exclude_subject=None, actor=None)`
  — ranks the gallery for one modality/provider/model. NumPy cosine path always
  available; `VECTOR_INDEX="pgvector"` requires the `biometric_pgvector` app
  (`ImproperlyConfigured` otherwise). With `actor` given and audit on, the
  ranking is recorded as an `identify` audit event.
- `fuse(scores, *, weights=None, thresholds=None, floors=None, floor_decision=None, required=frozenset())`
  — multimodal decision (`accept`/`review`/`reject`), tighten-only.
- `consolidate(subject_model=None, kept_id, retired_id, *, actor)` — re-points
  `retired`'s active templates to `kept` (or supersedes on key collision).
  Bound to the `deduplication.subject_merged` service signal.
- `templates_of(subject_model=None, subject_id, *, modality=None, actor, purpose="read")`
  — the only sanctioned plaintext read path; decrypts and logs the access.
- `purge(now=None, *, actor="retention")` — two independent, off-by-default
  passes: superseded templates past `template_retention_days` (when
  `purge_enabled`), then still-active templates past
  `active_template_retention_days` (when `purge_active_enabled`), tombstoned
  with `reason="ACTIVE_AGE"`. Run via the `biometric_purge` management command.

### GraphQL

Mutations `enrolBiometric`, `verifyBiometric`, `recordBiometricConsent`,
`acknowledgeBiometricAlert`, `resolveBiometricAlert`;
queries `identifyBiometric(modality, sample, topK, excludeSubject)`,
`biometricTemplates(subjectId, subjectModel)` (metadata only — never
plaintext vectors/templates), `biometricVerifications(subjectId, subjectModel)`,
and the relay connections `biometricAuditEvents` and `biometricAlerts`.
`subjectModel` is optional everywhere it appears, defaulting to
`BIOMETRIC["SUBJECT_MODEL"]`. Samples travel base64 (optionally with a
`data:...;base64,` prefix).

### Deduplication seam

This app registers a `BiometricCandidateSource` (kind `"biometric"`) with
`deduplication.sources` in `AppConfig.ready()`, guarded by
`try/except ImportError` — the deduplication package is optional. It scans
active templates for one modality, runs `identify()` against the rest of the
gallery, and yields a candidate per match at or above `DEDUP_THRESHOLD`.

---

## `biometric_pgvector` — pgvector ANN index (optional)

`biometric.services.identify()`'s default gallery search is one NumPy matrix
product over every active embedding template — exact, and fine up to a
gallery of a few hundred thousand rows. Past that, install `biometric_pgvector`
and set `BIOMETRIC["VECTOR_INDEX"] = "pgvector"` to search an
[HNSW](https://github.com/pgvector/pgvector#hnsw) approximate index in
Postgres instead. Without this app installed, `identify()` raises
`ImproperlyConfigured` as soon as `VECTOR_INDEX` is set to `"pgvector"`.

### When to install it

- A gallery large enough that the NumPy path's per-call full scan is too slow
  (deduplication scans and identify-on-enrol calls both re-run it).
- A Postgres server that has the `vector` extension available — the first
  migration runs `CREATE EXTENSION IF NOT EXISTS vector`, which needs
  superuser (or a role pre-granted `CREATEDB`/extension rights).
- **Not** a fit when `BIOMETRIC["TEMPLATE_KEY"]` is set and plaintext vectors
  must never leave the encrypted store — see the trade-off below.

### Quick start

```bash
pip install "openimis-be-biometric_verification[pgvector]"
```

Add to `openimis.json`, after `biometric`:
```json
{
  "modules": ["biometric", "biometric_pgvector"]
}
```

Run migrations (creates the `vector` extension and `biometric_vector_index`):
```bash
python manage.py migrate biometric_pgvector
```

Set `BIOMETRIC["VECTOR_INDEX"] = "pgvector"`.

### The two commands

- `python manage.py biometric_vector_reindex` — backfills
  `biometric_vector_index` from every `biometric_template` row (fresh install,
  or after data loaded without going through the ORM's signals). Safe to
  re-run; also drops side rows for templates that should no longer have one.
- `python manage.py biometric_vector_index --model NAME --dim N [--drop]` —
  creates (or, with `--drop`, removes) a partial HNSW index for one
  `model_name` at a fixed vector dimension:
  ```sql
  CREATE INDEX IF NOT EXISTS <name> ON biometric_vector_index
    USING hnsw ((embedding::vector(N)) vector_cosine_ops)
    WHERE model_name = 'NAME'
  ```
  Run once per `(model_name, dim)` pair actually enrolled — the cast and the
  `WHERE` clause must match `identify()`'s query exactly for the planner to
  use the index. `BIOMETRIC["HNSW_EF_SEARCH"]` (default `200`) controls the
  search-time accuracy/speed trade-off (`SET LOCAL hnsw.ef_search`).

Day to day, `biometric_vector_index` stays in sync on its own: `enrol()`,
`consolidate()`, and template deletion all fire `BiometricTemplate`
`post_save`/`post_delete`, which `biometric_pgvector` uses to upsert or drop
the corresponding side row — nothing needs to be re-run after normal use.

### The plaintext trade-off

**An ANN index cannot search encrypted vectors.** `biometric_vector_index`
always stores the embedding in clear, decrypted from `biometric_template` at
sync time, regardless of `BIOMETRIC["TEMPLATE_KEY"]`. If a key is set —
templates are meant to be encrypted at rest — installing this app would
silently defeat that guarantee, so `biometric_pgvector` refuses to start
(`ImproperlyConfigured` at `AppConfig.ready()`) unless
`BIOMETRIC["ALLOW_PLAINTEXT_INDEX"] = True` is set as an explicit,
deliberate opt-in. There is no partial middle ground: either encryption at
rest covers every stored vector, or this index's rows are the exception.

---

## `biometric_verification` — legacy 1:1 face verification (health)

Verifies an insuree's identity at point of service by comparing a live
webcam capture against the reference photo stored at enrollment. Requires
`insuree` and `claim` installed — `BiometricEmbedding` FKs to `insuree.Insuree`
and `ClaimFacialAudit` FKs to `claim.Claim` unconditionally.

### Quick start

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

### Usage

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

### Switching Providers

Change `PROVIDER` in `settings.py` — no code changes needed:

| Value | Description |
|---|---|
| `deepface` | Local inference via DeepFace (default) |
| `aws_rekognition` | AWS Rekognition cloud API |
| `azure_face` | Azure Face API |
| Custom | Any class implementing `BaseBiometricProvider` |

See `CLAUDE.md` for full provider implementation guide.

### Requirements

- Python 3.8+
- openIMIS backend (Django), `insuree` and `claim` installed
- `deepface`, `opencv-python-headless`, `tf-keras`

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

# Biometric identity and deduplication — implementation contract

Shared between `openimis-be-biometric-verification_py` (branch `feature/multimodal-identify`)
and `openimis-be-deduplication_py` (branch `feature/candidate-sources`). Both branches are
implemented against this document; where the document and an implementer disagree, the
document is corrected first, then the code.

Design rationale lives in the design note; this file is the executable contract.

## 0. Ground rules

- Target: the openIMIS assembly at `release/26.04` semantics — Django 4.2, **graphene 2.1.x**
  (not graphene 3), DRF 3.x, Postgres. Follow the conventions of the module being extended
  (`apps.py` `ModuleConfiguration` for rights, `core.signals` service signals, numeric rights).
- Everything is **additive**. Existing models, migrations, queries and mutations keep their
  behaviour. New tables get new migrations. No existing migration is edited.
- Comments in English, present tense, about the code they sit on. No history, no dates.
- Every new service has tests. Tests use fake providers; no ML model is downloaded in tests.
- The two modules must not import each other at module import time. Cross-module imports are
  inside functions or guarded by `try/except ImportError`, so each module installs alone.

## 1. Subject

Both modules address a **subject** by `(subject_model, subject_id)`:

- `subject_model`: dotted label, e.g. `"individual.Individual"` (social protection) or
  `"insuree.Insuree"` (health). Resolved with `django.apps.apps.get_model`.
- `subject_id`: the subject's primary key as a string (UUID for Individual).

Config: `BIOMETRIC["SUBJECT_MODEL"]` and `DEDUPLICATION["SUBJECT_MODEL"]`, both defaulting to
`"individual.Individual"`. No `GenericForeignKey`/contenttypes dependency: two plain columns.

## 2. The seam — two things cross it, nothing else

### 2.1 `deduplication.sources` (owned by the deduplication module)

```python
@dataclass(frozen=True)
class Watermark:
    updated_at: datetime | None = None
    last_id: str | None = None

@dataclass(frozen=True)
class Candidate:
    subject_model: str
    subject_a: str          # ordered: subject_a < subject_b as strings
    subject_b: str
    kind: str               # "demographic" | "identifier" | "biometric" | any registered kind
    score: float | None     # higher = more likely the same person; None for exact matches
    evidence: dict          # JSON-serialisable; what a reviewer needs to see

class CandidateSource(abc.ABC):
    kind: str
    @abc.abstractmethod
    def scan(self, since: Watermark | None) -> Iterable[Candidate]: ...
    @abc.abstractmethod
    def watermark(self) -> Watermark: ...      # where the next scan starts

def register(source: CandidateSource) -> None
def sources() -> list[CandidateSource]        # registration order
def order_pair(a: str, b: str) -> tuple[str, str]
```

The biometric module registers its source in `AppConfig.ready()`:

```python
try:
    from deduplication.sources import register
except ImportError:
    return
register(BiometricCandidateSource())
```

### 2.2 Service signal `deduplication.subject_merged` (emitted by deduplication)

Registered and emitted with the same `core.signals` mechanism the deduplication module already
uses for `task_service.complete_task` (read `deduplication/signals/__init__.py` and
`core/signals.py` for the exact API — `register_service_signal` / `bind_service_signal`,
`ServiceSignalBindType`). Payload kwargs:

```
subject_model: str, kept_id: str, retired_id: str, actor: str, policy: "retire" | "delete"
```

The biometric module binds `AFTER` and calls `consolidate(kept, retired)` (§3.6).

## 3. Biometric module — `feature/multimodal-identify`

Package name stays `biometric_verification` (the rename is a community decision, out of scope).

### 3.1 Providers — `providers/base.py`

Keep `BaseBiometricProvider` and `VerificationResult` exactly as they are (backward
compatibility for the insuree 1:1 flow). Add two provider kinds:

```python
@dataclass
class Extracted:
    vector: list[float] | None = None     # EmbeddingProvider
    template: bytes | None = None         # MatcherProvider, vendor format
    template_iso: bytes | None = None     # ISO/IEC 19794 where the SDK gives it
    quality: float | None = None          # 0–100, NFIQ-like where applicable
    metadata: dict = field(default_factory=dict)

class ModalityProvider(abc.ABC):
    modality: str            # "face" | "fingerprint" | "voice" | "iris" | "palmvein"
    provider_name: str
    kind: str                # "embedding" | "template"
    default_threshold: float
    @abc.abstractmethod
    def extract(self, sample: bytes, position: str | None = None) -> Extracted: ...
    def health_check(self) -> bool: return True

class EmbeddingProvider(ModalityProvider):
    kind = "embedding"
    @abc.abstractmethod
    def distance(self, a: list[float], b: list[float]) -> float: ...   # lower = closer
    def similarity(self, a, b) -> float: return 1.0 - self.distance(a, b)

class MatcherProvider(ModalityProvider):
    kind = "template"
    @abc.abstractmethod
    def match(self, probe: bytes, reference: bytes) -> float: ...   # higher = more similar
```

Scores exposed to callers are always **similarity, higher = more similar**, on the provider's
own scale; each provider carries its `default_threshold` on that scale.

Providers shipped:
- `deepface_provider.DeepFaceProvider` — adapted to implement `EmbeddingProvider` for
  `modality="face"` while still satisfying `BaseBiometricProvider` (the existing class gains the
  new methods; `extract` wraps `get_embedding`; `distance` is the existing cosine).
- `providers/device_reported.py` — `DeviceReportedMatcher(modality)`: a `MatcherProvider` whose
  `extract` stores the bytes as given and whose `match` raises `NotImplementedError`; it exists
  so a deployment where matching happens on the device (fingerprint on a tablet) still has a
  registered provider that stores templates and owns the threshold. Verification for such a
  modality goes through the device-reported path (§3.5).
- `providers/fake.py` — `FakeEmbeddingProvider`, `FakeMatcherProvider` for tests only
  (deterministic: vector = bytes hashed to floats; match = 100 if equal else 0).

### 3.2 Registry — `registry.py`

Keep `ProviderRegistry.register` / `get_active_provider` (face, legacy). Add:

```python
ProviderRegistry.register_modality(modality: str, name: str, cls: type[ModalityProvider])
ProviderRegistry.get_provider(modality: str) -> ModalityProvider   # cached instance
```

Config, read by `BiometricVerificationConfig` from `BIOMETRIC_VERIFICATION` / module
configuration, all optional with defaults:

```python
"MODALITIES": {
    "face":        {"provider": "deepface",        "threshold": 0.68},   # legacy default kept
    "fingerprint": {"provider": "device_reported", "threshold": 48},
},
"VECTOR_INDEX": "numpy",          # "numpy" | "pgvector"  (pgvector only if importable)
"TEMPLATE_KEY": None,             # Fernet key; templates and vectors are encrypted at rest when set
"REQUIRE_CONSENT": False,
"DEDUP_THRESHOLD": {"face": 0.62},  # similarity at/above which a candidate is emitted
"FUSION": {"weights": {"face": 1.0}, "thresholds": {"accept": 0.7, "review": 0.6},
           "floors": {}, "floor_decision": "review"},
```

### 3.3 Models — new migrations only (`0005_…` onward); existing tables untouched

`BiometricTemplate` (`db_table="biometric_template"`)
- `id` UUID pk; `subject_model` char(64); `subject_id` char(64) indexed
- `modality` char(16); `position` char(16) blank (`"R_THUMB"`, `"L_INDEX"`, `"R_IRIS"`, …)
- `kind` char(16) `"embedding"|"template"`
- `vector` JSON null; `template` binary null; `template_iso` binary null
- `encrypted` bool — when `TEMPLATE_KEY` is set, `vector` is stored as an encrypted string and
  `template*` as encrypted bytes (Fernet from `cryptography`); when unset, stored plain and the
  app logs one warning at startup. Encryption/decryption lives in `crypto.py`; models never
  expose plaintext by accident — access goes through `services.templates_of()` which decrypts
  and writes an access log entry.
- `quality` float null; `provider` char(64); `model_name` char(64); `metadata` JSON
  (scope keys such as `{"cuvee_id": …}` live here)
- `validity_from` auto_now_add; `validity_to` null; `date_created`; `date_updated`
- Unique among active rows: `(subject_model, subject_id, modality, position, provider, model_name)`
  where `validity_to IS NULL`; index `(modality, provider, model_name, validity_to)`.

`BiometricVerification` (`db_table="biometric_verification"`) — the generic audit
- subject ref; `modality`; `score` float null; `threshold` float; `verified` bool
- `origin` `"server"|"device"`; `fallback` bool; `context` JSON; `device_id` char(255) blank
- `actor` char(64); `created_at`. Never stores a sample. `ClaimFacialAudit` stays as is.

`BiometricConsent` — subject ref; `modality`; `granted` bool; `recorded_by`; `recorded_at`; `note`.

`BiometricRetentionPolicy` — singleton: `template_retention_days` int null;
`purge_enabled` bool; a check constraint requires both to be set for a purge to act.

`BiometricErasure` — tombstone: subject ref (bare strings, no FK), `modalities` JSON,
`erased` JSON counts, `reason` char(32), `erased_by`, `erased_at`.

`BiometricAccessLog` — subject ref; `actor`; `purpose` char(32); `template_ids` JSON; `at`.

### 3.4 Services — `services.py` (existing functions kept)

```python
enrol(subject_model, subject_id, modality, sample: bytes, *, position=None, actor, metadata=None,
      device_template: Extracted | None = None) -> BiometricTemplate
```
Refuses when `REQUIRE_CONSENT` and no granted consent for the modality. With a `MatcherProvider`
of kind `device_reported`, `device_template` is stored as given (the device extracted it).
Supersedes an active row with the same unique key (`validity_to = now`) rather than updating it.

```python
verify(subject_model, subject_id, modality, *, sample: bytes | None = None, position=None,
       device_score: float | None = None, fallback=False, context=None, device_id="", actor)
       -> VerificationResult   # extended with .modality, .origin, .threshold
```
Server path: extract, compare with every active template of the subject for that modality
(and position when given), keep the best similarity. Device path: `device_score` is checked
against the modality threshold; nothing is extracted. Both record a `BiometricVerification`.

```python
identify(modality, *, sample: bytes | None = None, vector=None, template=None, top_k=5,
         scope: dict | None = None, exclude_subject: str | None = None) -> list[Match]
# Match(subject_model, subject_id, template_id, score)
```
Gallery = active templates for the modality's configured provider and model, filtered by
`scope` on `metadata` keys. Embedding kind: one NumPy matrix product (`gallery @ probe` after
L2 normalisation) — the portable path, always available. `VECTOR_INDEX == "pgvector"` with the
package importable: an `ORDER BY vector <=> probe LIMIT top_k` path (implement behind the
flag; tests cover the NumPy path and assert the flag routes). Template kind: iterate `match()`.

```python
fuse(scores: dict[str, float | None], *, weights=None, thresholds=None, floors=None,
     floor_decision=None, required: set[str] = frozenset()) -> Decision
# Decision(outcome: "accept"|"review"|"reject", score: float | None, reasons: list[str])
```
Weighted mean of present, positively-weighted legs, normalised to the provider thresholds
(a leg's score is divided by its modality threshold so 1.0 = at threshold). Rules only ever
**tighten**: a `None` leg that is `required` → `review`; a leg below its floor → `floor_decision`;
score ≥ accept → accept, ≥ review → review, else reject.

```python
consolidate(subject_model, kept_id, retired_id, *, actor) -> dict   # counts per modality
```
Re-points active templates of `retired` to `kept`; where `kept` already holds an active row with
the same `(modality, position, provider, model_name)`, the retired row is superseded instead.
Writes an access log entry. Bound to `deduplication.subject_merged`.

```python
purge(now=None, *, actor="retention") -> BiometricErasure | None
```
Management command `biometric_purge`. Acts only when the policy has both fields set.

### 3.5 Candidate source — `dedup_source.py`

`BiometricCandidateSource(modality="face")`: `scan(since)` iterates active templates of the
modality whose `date_updated`/id are past the watermark, runs `identify(top_k, exclude_subject)`
for each, and yields a `Candidate(kind="biometric", score, evidence={"modality", "provider",
"model_name", "template_a", "template_b"})` per match at or above `DEDUP_THRESHOLD[modality]`.
Pairs are ordered with `order_pair`. `watermark()` returns the newest `(date_updated, id)` seen.

### 3.6 GraphQL — graphene 2, `schema.py` (existing fields kept)

Mutations `enrolBiometric`, `verifyBiometric`, `recordBiometricConsent`; queries
`identifyBiometric`, `biometricTemplates(subjectModel, subjectId)`,
`biometricVerifications(subjectModel, subjectId)`. Rights via module configuration, defaults
`gql_biometric_enrol_perms=["174001"]`, `gql_biometric_verify_perms=["174002"]`,
`gql_biometric_identify_perms=["174003"]`, `gql_biometric_read_perms=["174004"]`, following
the pattern of the module's existing rights. Samples travel base64.

### 3.7 Tests (pytest, fake providers)
crypto round-trip; enrol (supersede, consent refusal, device template); verify server and device
paths with audit rows; identify NumPy ranking, scope filter, self-exclusion, flag routing;
fuse (weighted mean, required leg, floor, bands, tighten-only); consolidate (re-point, supersede
collision, signal binding); candidate source (threshold, ordering, watermark); purge (policy
guard, tombstone); registry per modality; legacy `get_active_provider` unchanged.

## 4. Deduplication module — `feature/candidate-sources`

### 4.1 `sources/` package — §2.1, plus two built-in sources

`DemographicSource`: `GROUP BY` on the subject model over configured columns
`DEDUPLICATION["DEMOGRAPHIC_COLUMNS"]` (default `["first_name", "last_name", "dob"]`; a name
not on the model is read as a `json_ext` key, reusing the column-resolution logic already in
`services.py:346-364`). Each group of n>1 yields all pairs, `score=None`,
`evidence={"columns": {...values...}}`. Watermark on `date_updated`/id of the subject model.

`IdentifierSource`: exact match on `DEDUPLICATION["IDENTIFIER_KEYS"]` (json_ext keys, default
`[]`), normalised (strip, casefold). Same shape.

The existing beneficiary/payment summary queries and their task flow are **untouched**.

### 4.2 Models — new migrations only (`0002_…` onward)

`DuplicateCandidate` (`db_table="deduplication_candidate"`, plain `models.Model`, uuid pk)
- `subject_model`; `subject_a`; `subject_b` (strings, `subject_a < subject_b` — CheckConstraint)
- `kind` char(32); `source` char(64); `score` float null; `evidence` JSON
- `status` `"OPEN"|"CONFIRMED"|"DISMISSED"`; `task` FK `tasks_management.Task` null
- `reviewed_by` char(64) blank; `reviewed_at` null; `decision_note` text blank
- `date_created`; `date_updated`
- unique `(subject_model, subject_a, subject_b, kind)`; index `(status, kind)`.

`ScanState` — one row per `kind`: `updated_at`, `last_id`, `last_scan_at`, `summary` JSON.

### 4.3 Services — `services.py` (existing functions kept)

```python
record_candidate(c: Candidate, *, source: str) -> tuple[DuplicateCandidate, bool]
```
`get_or_create` on the unique key. A `DISMISSED` row is never reopened. An `OPEN` row keeps the
higher score and merges evidence. Returns `(row, created)`.

```python
run_scan(*, kinds: list[str] | None = None, actor: str) -> dict   # counts per kind
scan_subject(subject_model, subject_id) -> list[DuplicateCandidate]   # on-demand, all sources
```
Management command `scan_duplicates [--kind KIND]`. `run_scan` reads `ScanState`, calls
`source.scan(since)`, records, then stores `source.watermark()`.

```python
resolve(candidate, *, decision: "same"|"different", keep: str | None = None, actor, note="")
```
`different` → `DISMISSED`. `same` → `CONFIRMED`, `keep` defaults to `subject_a`, then
`merge_subjects(kept, retired, actor)`:

- `DEDUPLICATION["MERGE_POLICY"]`: `"delete"` (default, legacy) or `"retire"`.
- Field policy on the subject model for both policies: an empty field on `kept` is filled from
  `retired`; a differing non-empty value is **kept and journaled** into
  `kept.json_ext["merge_conflicts"]` as `{"field", "kept", "retired", "retired_id", "at", "actor"}`.
  Never overwrite. `json_ext` keys are merged the same way.
- `retire`: `retired.json_ext["retired_into"] = kept_id`, then the model's own soft delete
  (`.delete(user=…)` on a `HistoryModel`). `delete`: the model's soft delete only.
- Emit `deduplication.subject_merged` (§2.2) **after** the transaction commits.

Review through Tasks Management stays available: `create_review_tasks(candidate_ids, actor)`
creates one `tasks_management.Task` per candidate (`source="deduplication_candidate"`, `data` =
candidate summary, `task` FK set); the existing `task_service.complete_task` binding gains a
branch: when the completed task's source is `deduplication_candidate`, call `resolve()` with the
decision read from `task.json_ext["additional_resolve_data"]` (`{"decision", "keep", "note"}`).

### 4.4 GraphQL — graphene 2 (existing fields kept)

Query `duplicateCandidates(status, kind, subjectId, first, offset)` following the connection
style the module already uses. Mutations `runDuplicateScan(kinds)`,
`resolveDuplicateCandidate(id, decision, keep, note)`, `createDuplicateReviewTasks(ids)`.
Rights: `gql_resolve_duplicate_perms=["172003"]`, `gql_run_scan_perms=["172004"]`,
`gql_query_duplicates_perms=["172005"]` via module configuration; review-task creation keeps
`172001`.

### 4.5 Tests
registry and `order_pair`; demographic source on fixture individuals (json_ext column);
identifier source normalisation; `record_candidate` idempotence and dismissed memory;
`run_scan` watermark advance and second-run no-op; `resolve` different/same; `merge_subjects`
fill-empty, conflict journal, both policies, signal emitted once after commit; task bridge;
GraphQL smoke for each field; management command.

## 5. Running the tests locally

Workspace `/Users/anthbel/projects/wb/cameroun`. Install the fork over the pinned module:
`.venv-cameroun/bin/pip install -e <fork path>` (deduplication replaces the `release/26.04`
pin; biometric_verification is new — add
`{"name": "biometric_verification", "pip": "-e <fork path>"}` to the **local**
`openimis-be_py/openimis.json` modules list). Then, from `openimis-be_py/openIMIS`:

```
DB_DEFAULT=postgresql DB_HOST=localhost DB_PORT=55432 DB_NAME=openimis DB_USER=openimisuser \
DB_PASSWORD=change-me SITE_ROOT=api OPENIMIS_CONF=<workspace>/openimis-be_py/openimis.json \
ASYNC=SYNC CELERY_TASK_ALWAYS_EAGER=True CELERY_BROKER_URL=memory:// \
<workspace>/.venv-cameroun/bin/pytest <fork path>/<package>/tests -v --no-migrations --reuse-db
```
The database is the `cameroun-db` container (`make db-up` in `openimis-dist-cameroun` starts it;
Docker Desktop must be running). New tables need the migrations applied once to the test DB:
run `<workspace>/.venv-cameroun/bin/python manage.py migrate <app>` from `openimis-be_py/openIMIS`
with the same environment, or drop `--no-migrations` for the first run.

## 6. Revision 2 — gaps closed

This section supersedes earlier sections where they conflict.

### 6.1 Generic app `biometric`; `biometric_verification` returns to upstream

The generic multimodal code moves to a **new Django app `biometric`** (package `biometric/`,
app label `biometric`) in the same repository and distribution. `biometric_verification`
goes back to the exact bytes of `upstream/develop` for every non-test file: its models,
its migrations 0001–0004 (unconditional again), providers, registry, services, schema,
consumers, routing. It is the health-domain app, bound to `insuree`/`claim`, installed
only by assemblies that have them. No conditional migration anywhere.

- Everything §3 specified lives in `biometric/`: `providers/` (ModalityProvider,
  EmbeddingProvider, MatcherProvider, DeepFace face provider, DeviceReportedMatcher, fakes),
  `registry.py` (per-modality registry only), `crypto.py`, `models.py` (the six tables,
  same `db_table` names), `services.py`, `dedup_source.py`, `signals.py`, `schema.py`
  (the six GraphQL fields, rights 174001–174004), `management/commands/biometric_purge.py`,
  `apps.py` (`BiometricConfig`, config key `BIOMETRIC`).
- `biometric` never imports `biometric_verification`. `biometric_verification` may later
  delegate to `biometric`; not in this revision.
- The face provider in `biometric` is its own `EmbeddingProvider` on the **similarity**
  scale; its threshold is configured as a similarity (`MODALITIES["face"]["threshold"]`,
  default 0.32 = legacy distance 0.68). Keep the agreement test against the legacy
  `biometric_verification` provider only where that app is installed (health env, 6.4).
- Migrations: `biometric/migrations/0001_initial.py` creates the six tables fresh.
- `setup.py`: `packages=find_packages()` already covers both; add
  `extras_require={"pgvector": ["pgvector>=0.3"]}`.

### 6.2 Optional app `biometric_pgvector`

Separate app (package `biometric_pgvector/`, label `biometric_pgvector`), installed only
where the Postgres server has the `vector` extension.

- Model `BiometricVectorIndex` (`db_table="biometric_vector_index"`): `template` OneToOne to
  `biometric.BiometricTemplate` (CASCADE, pk), `modality`, `provider`, `model_name`,
  `dim` int, `embedding` = `pgvector.django.VectorField()` without fixed dimensions.
  Migration 0001 runs `pgvector.django.VectorExtension()` then creates the table.
- Kept in sync by signal receivers on `BiometricTemplate` (post_save / post_delete): an active
  embedding-kind row gets its side row upserted from the decrypted vector; a superseded or
  deleted row loses it. Management command `biometric_vector_reindex` backfills.
- Per-model HNSW index, dimension-specific, created by management command
  `biometric_vector_index --model NAME --dim N [--drop]`:
  `CREATE INDEX IF NOT EXISTS <name> ON biometric_vector_index USING hnsw
  ((embedding::vector(N)) vector_cosine_ops) WHERE model_name = 'NAME'`.
- `biometric.services.identify` with `VECTOR_INDEX="pgvector"`: raise
  `ImproperlyConfigured` unless `biometric_pgvector` is installed; otherwise query the side
  table with the same cast expression (`ORDER BY (embedding::vector(N)) <=> %s::vector(N)
  LIMIT k`), inside a transaction that sets `SET LOCAL hnsw.ef_search = <config, default
  200>`, applying the `scope` filter through the template join and `exclude_subject`.
  Returns the same `Match` objects as the NumPy path, similarity = 1 − cosine distance.
- Plaintext: the index stores vectors in clear — an ANN index cannot search encrypted
  vectors. When `TEMPLATE_KEY` is set, `biometric_pgvector` refuses to start
  (`ImproperlyConfigured`) unless `BIOMETRIC["ALLOW_PLAINTEXT_INDEX"]` is `True`. README
  states this trade-off plainly.
- Tests run against a vector-capable Postgres: container `pgvector/pgvector:pg13` on
  port 55433 (same major as dev), test DB prepared exactly like `test_imis`
  (`openimis-dist-cameroun/scripts/setup_test_db.sh` with the port/name overridden).
  Required cases: side row sync (create, supersede, delete, consolidate), reindex,
  index command creates the named index, `identify` pgvector path returns the same top-k
  as the NumPy path on the same gallery, scope and self-exclusion, plaintext guard.

### 6.3 Behaviour options

- Purge: `BiometricRetentionPolicy` gains `active_template_retention_days` (null) and
  `purge_active_enabled` (bool, default False). When both are set, **active** templates whose
  `validity_from` is older than the window are erased too (after superseded ones), each
  subject getting a tombstone with `reason="ACTIVE_AGE"`. Check constraint: enabled requires
  the window. Default behaviour unchanged.
- `subject_model`: every service and GraphQL argument `subject_model` becomes optional and
  defaults to `BIOMETRIC["SUBJECT_MODEL"]` (`"individual.Individual"`).
- Deduplication `IdentifierSource`: config `DEDUPLICATION["IDENTIFIER_MATCH"]` =
  `"each"` (default, current behaviour: any single key matching) or `"all"` (compound: all
  configured keys equal and non-empty). Tests for both.

### 6.4 Legacy tests of `biometric_verification`

Fix the 23 upstream tests that fail on `upstream/develop`; never weaken an assertion.
- Tests patching an attribute where it is not looked up: patch where it is looked up
  (e.g. `biometric_verification.apps.BiometricVerificationConfig.<attr>` via
  `patch.object`, or the lazily imported module path).
- `VerifyFaceMutation` tests: call with the mutation's real argument names (`uuid`,
  `frame`) — the GraphQL API is the contract; the tests follow it.
- Tests needing `insuree`/`claim`: run in a **health test environment** — a separate venv
  `/Users/anthbel/projects/wb/cameroun/.venv-health` and manifest
  `/Users/anthbel/projects/wb/cameroun/openimis-health.json` holding the smallest
  import-closed set of upstream modules (`release/26.04`) that makes `insuree` and `claim`
  install and migrate, plus `biometric_verification` (-e). Record the exact set and the
  run command in §5.

### 6.5 Local manifests

The shared `openimis-be_py/openimis.json` goes back to the synced state (no
`biometric_verification` line) so `make check-manifests` / `make test-be` pass. Fork testing
uses its own manifest `/Users/anthbel/projects/wb/cameroun/openimis-forks.json` = the synced
manifest + `{"name": "biometric", "pip": "-e <repo>"}` (and `biometric_pgvector` for the
pgvector run), passed via `OPENIMIS_CONF`. The test env always includes
`MODE=dev DJANGO_SETTINGS_MODULE=openIMIS.settings`.

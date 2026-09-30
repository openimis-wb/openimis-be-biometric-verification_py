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
"RISK_PROFILES": {},              # named tighten-only overrides of FUSION / MODALITIES (§6.8)
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
  Every reader decrypts a row by its `encrypted` flag (`crypto.row_key`): `encrypted=False` is
  read as stored; `encrypted=True` is decrypted with `TEMPLATE_KEY`. A row marked encrypted that
  does not decrypt (wrong or rotated key, or no key set) raises `crypto.TemplateKeyError`, whose
  text never quotes the stored value and whose GraphQL error carries
  `extensions.code = "BIOMETRIC_TEMPLATE_KEY"`. `verify()` logs the template id at ERROR and
  raises before writing its `BiometricVerification` row, so a key fault is an error, never
  `verified=false`; `identify()`, `templates_of()`, the candidate scan and the
  `biometric_pgvector` sync receiver raise the same error.
- `quality` float null; `provider` char(64); `model_name` char(64); `metadata` JSON
  (scope keys such as `{"cuvee_id": …}` live here)
- `validity_from` auto_now_add; `validity_to` null; `date_created`; `date_updated`
- Unique among active rows: `(subject_model, subject_id, modality, position, provider, model_name)`
  where `validity_to IS NULL`; index `(modality, provider, model_name, validity_to)`.

`BiometricVerification` (`db_table="biometric_verification"`) — the generic audit
- subject ref; `modality`; `score` float null; `threshold` float; `verified` bool
- `origin` `"server"|"device"`; `fallback` bool; `context` JSON; `device_id` char(255) blank
- `risk_profile` char(64) blank, default `""`: the named risk profile the threshold was resolved
  under (§6.8); `threshold` holds the effective value after the profile.
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
       device_score: float | None = None, fallback=False, context=None, device_id="", actor,
       risk_profile: str | None = None, device_template: Extracted | None = None)
       -> VerificationResult   # extended with .modality, .origin, .threshold, .risk_profile,
                               # .impersonation, .impersonation_skip_reason
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
     floor_decision=None, required: set[str] = frozenset(), risk_profile: str | None = None) -> Decision
# Decision(outcome: "accept"|"review"|"reject", score: float | None, reasons: list[str],
#          risk_profile: str = "")
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
`verifyBiometric` takes an optional `riskProfile: String` (§6.8); `BiometricVerifyResultType`
and `BiometricVerificationRecordType` expose `riskProfile: String`. Later additions:
`verifyBiometricMultimodal` (§6.11), `deviceVector` / `deviceTemplate` and
`impersonationSkipReason` (§6.9), and the admin fields of §6.12.

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
Management command `scan_duplicates [--kind KIND]`. `run_scan` reads `ScanState`, captures
`source.watermark()`, calls `source.scan(since)`, records, then stores the captured watermark.
A subject written while the scan iterates is newer than the captured cursor and is revisited by
the next scan; `record_candidate` is idempotent, so the revisit is harmless.

```python
resolve(candidate, *, decision: "same"|"different", keep: str | None = None, actor, note="")
```
`different` → `DISMISSED`. `same` → `CONFIRMED`, `keep` defaults to `subject_a`, then
`merge_subjects(kept, retired, actor)`:

- Precondition: the subject to retire holds no live enrolment row. While it still has a
  non-deleted `social_protection.Beneficiary`, a non-deleted `individual.GroupIndividual`, or a
  non-deleted `social_protection.GroupBeneficiary` on one of its groups, `resolve` and
  `merge_subjects` raise `ValueError("deduplication.resolve.retired_subject_enrolled: …")` naming
  the blocking rows by kind and count, before any write; the candidate stays `OPEN`. The rows are
  removed first (soft delete), then the merge is retried. Payroll selects active beneficiaries
  without reading the individual, so a soft-deleted individual with a live row stays payable.
  The models are resolved by `apps.get_model`; an absent app blocks nothing. Rows on `kept` never
  block.
- `DEDUPLICATION["MERGE_POLICY"]`: `"delete"` (default, legacy) or `"retire"`.
- Field policy on the subject model for both policies: an empty field on `kept` is filled from
  `retired`; a differing non-empty value is **kept and journaled** into
  `kept.json_ext["merge_conflicts"]` as `{"field", "kept", "retired", "retired_id", "at", "actor"}`.
  Never overwrite. `json_ext` keys are merged the same way.
- `retire`: `retired.json_ext["retired_into"] = kept_id`, then the model's own soft delete
  (`.delete(user=…)` on a `HistoryModel`). `delete`: the model's soft delete only.
- Emit `deduplication.subject_merged` (§2.2) **after** the transaction commits.

A pair proposed by several sources holds one candidate per `kind`; the merge settles them
together. In the merge transaction every other `OPEN` candidate on the same unordered pair becomes
`CONFIRMED` with the same decision, and a `DISMISSED` one stays dismissed. A `same` on a candidate
whose pair already has a `CONFIRMED` sibling becomes `CONFIRMED` without a second merge when `keep`
is the surviving subject and the other subject is deleted; a `keep` naming the deleted subject is
refused with `deduplication.resolve.keep_contradicts_merge`.

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

### 6.6 Test environments (as built and verified)

Common test env for every run below, from `openimis-be_py/openIMIS`:
`MODE=dev DJANGO_SETTINGS_MODULE=openIMIS.settings DB_DEFAULT=postgresql DB_HOST=localhost
DB_USER=openimisuser DB_PASSWORD=change-me SITE_ROOT=api ASYNC=SYNC
CELERY_TASK_ALWAYS_EAGER=True CELERY_BROKER_URL=memory:// SCHEDULER_AUTOSTART=`, then
`<venv>/bin/pytest <paths> --reuse-db`. The test database name resolves to `test_imis`
whatever `DB_NAME` says (`openIMIS/settings/database.py` defaults `DB_TEST_NAME` that way).

| Run | Port / container | Manifest | Venv | Result |
|---|---|---|---|---|
| `biometric`, `deduplication` | 55432 / `openimis-dist-cameroun-postgres-1` (PG 13) | `openimis-forks.json` | `.venv-cameroun` | biometric 106 + 1 skip; dedup 30 |
| `biometric_pgvector`, `biometric` | 55434 / `biometric-pgvector-test` (`pgvector/pgvector:pg13`) | `openimis-forks-pgvector.json` | `.venv-cameroun` + `pgvector` | 20; 106 + 1 skip |
| `biometric_verification` (health) | 55435 / `biometric-health-db` (`openimis-pgsql:25.10`) | `openimis-health.json` | `.venv-health` | 47 |

Port 55433 belongs to an unrelated project's database; never use it.

**pgvector test DB.** The throwaway image has no openIMIS schema, so it is cloned read-only
from the dev DB: create `test_imis` on 55434; `pg_dump --schema-only --no-owner
--no-privileges` of 55432/`openimis` piped through `grep -v postgres-json-schema` (an
extension the pgvector image lacks, unused) into it; `pg_dump --data-only` of the six
bootstrap tables `setup_test_db.sh` copies; run `drop_orphan_fks.py`; then
`manage.py migrate biometric_pgvector` (creates the `vector` extension and the side table).

**Health venv and DB.** Modules at `release/26.04`: core, location, medical,
medical_pricelist, product, payer, insuree, calculation, contribution_plan, policy,
contribution, invoice, claim_batch, claim, report — the smallest set for which `insuree` and
`claim` migrate and `manage.py check` passes (`policy` imports `contribution_plan`, whose
migrations depend on `calculation`; `claim_batch` imports `contribution` and `invoice`;
`claim.views` imports `report`; every manifest entry's `urls` is included, so `check`
walks them all) — plus `biometric` and `biometric_verification` (`-e` this repo).
Build: `python3.11 -m venv .venv-health`; `pip install -r openimis-be_py/requirements.txt
-c openimis-dist-cameroun/constraints.txt`; each module cloned `--depth 1 -b release/26.04`
with retries and installed `-e` with `--retries 10 --timeout 60`; `pytest==9.0.3
pytest-django==4.12.0`. DB: `manage.py migrate` (`NO_DATABASE=True`), `manage.py check`,
`openimis-dist-cameroun/scripts/setup_test_db.sh` with `DB_PORT=55435
DB_NAME=openimis_health`, then `drop_orphan_fks.py` against `test_imis`.

### 6.7 Enrolment quality gate

`biometric/quality.py` judges every sample `enrol()` stores. It is provider-agnostic, needs
only NumPy, and reads images through Pillow when Pillow is importable (it is not a
dependency). It runs after extraction and before encryption and the supersede loop.

**Face geometry.** Providers may attach `Extracted.face: FaceGeometry | None` (last field of
`Extracted`, default `None`):

```python
@dataclass
class FaceGeometry:
    box: tuple[float, float, float, float] | None = None   # x, y, width, height
    landmarks: dict[str, tuple[float, float]] = {}          # (x, y) per name
    pose: dict[str, float] | None = None                    # degrees: yaw, pitch, roll
```

- Pixel grid: the sample as Pillow decodes it, without EXIF transpose. Origin top-left,
  x to the right, y downwards.
- Landmark names read: `left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right`; other
  keys are ignored.
- Pose: any subset of `yaw`, `pitch`, `roll`, in degrees; positive roll turns the head
  clockwise in the image.
- Geometry is never persisted: neither `metadata` nor the verdict carries a box or a
  landmark, only measures derived from them.

**Measures.** Each measure records `name`, `value`, `limit`, `kind` (`min` | `max`),
`passed`, `source` (`image` | `provider_pose` | `landmarks` | `provider`) and `detail`.
A measure is judged only when both value and limit are non-null; `passed` is `null`
otherwise. The bound itself is admitted (`value >= limit`, `abs(value) <= limit`).

| Modality | Measure | Limit key | Default |
|---|---|---|---|
| face | `sharpness` — variance of the 5-point Laplacian of the whole grayscale sample | `min_sharpness` | 100.0 |
| face | `lower_face_uniformity` — median summed absolute CIE L*a*b* deviation from the region's median colour, inside the lower part of the provider box, mouth band cut out | `min_lower_face_uniformity` | None |
| face | `yaw`, `pitch`, `roll` — provider pose when given; else roll from the eye line | `max_yaw`, `max_pitch`, `max_roll` | 20.0, None, 20.0 |
| face | `yaw_ratio` — nose offset along the eye axis over half the inter-ocular distance (landmarks only) | `max_yaw_ratio` | None |
| any | `provider_quality` — `Extracted.quality` | `min_quality` | None |

Other modalities carry `provider_quality` only.

**Verdict.** Stored on `BiometricTemplate.quality_verdict` (JSON, nullable; `NULL` = the gate
never ran on that row) and exposed as `BiometricTemplateType.qualityVerdict`:

```
{"status": "ACCEPTED" | "REFUSED" | "NOT_ASSESSED", "mode": "advisory" | "enforce",
 "modality": str, "reasons": [str], "measures": [measure], "version": 1}
```

- `REFUSED` when any judged measure fails; `ACCEPTED` when at least one is judged and none
  fails; `NOT_ASSESSED` when nothing was judged.
- Reasons, one per failed judged measure, in measure order: `sharpness_below_min`,
  `lower_face_uniformity_below_min`, `yaw_above_max`, `pitch_above_max`, `roll_above_max`,
  `yaw_ratio_above_max`, `provider_quality_below_min`; plus `sample_undecodable`.
- Server extraction (no `device_template`) of a sample Pillow cannot decode: `REFUSED`,
  reason `sample_undecodable`. With a `device_template`, image measures of an undecodable
  sample carry detail `sample_not_image` and only the device's `quality` and `face` are
  judged.
- Pillow absent: image measures carry detail `pillow_unavailable` and never refuse.

**Config.** `BIOMETRIC["QUALITY"]` / `quality` in the `biometric` module configuration:

```python
"QUALITY": {
    "mode": "advisory",            # "advisory" | "enforce"
    "modalities": {
        "face": {"min_sharpness": 100.0, "max_yaw": 20.0, "max_pitch": None, "max_roll": 20.0,
                 "max_yaw_ratio": None, "min_lower_face_uniformity": None, "min_quality": None},
    },
},
```

Nested keys are lowercase. Thresholds merge per key over the built-in defaults, so a partial
dict keeps the others. Any modality not listed uses `{"min_quality": None}`. An unknown mode
is logged at startup and raises `ImproperlyConfigured` when `enrol()` runs.

**Modes.**
- `advisory` (default): the verdict is stored; everything else `enrol()` does is unchanged,
  and the row joins the gallery whatever its status.
- `enforce`: a `REFUSED` verdict raises `QualityRefusedError(verdict)` (a `ValueError`)
  before anything is superseded or written; the subject's previous active template stays
  active. `NOT_ASSESSED` is never refused.

**GraphQL error.** In enforce mode a refused `enrolBiometric` returns `data.enrolBiometric:
null` and `errors[0].extensions = {"code": "BIOMETRIC_QUALITY_REFUSED", "verdict": {...}}`.
The message lists the reason codes and never the subject id.

**Refusal trace.** With audit on (§6.10), a refused enrolment records a
`template.enrol_refused` event carrying the verdict (status, mode, reasons, measures) and the
provider metadata, never the sample or the vector. No template row is written.

**DeepFace geometry.** `DeepFaceProvider.extract()` returns the embedding and
`face_geometry_from_deepface(result, detector_backend=...)` of the same `DeepFace.represent()`
result:

- `box` = `facial_area` `x`, `y`, `w`, `h`, dropped unless all four are finite numbers with a
  positive width and height.
- `landmarks` = the `facial_area` points `left_eye`, `right_eye`, `nose`, `mouth_left`,
  `mouth_right` that are present and finite. DeepFace returns the eyes on every supported version
  (None when not found). It returns nose and mouth corners only on versions that pass them through,
  and only for detectors that report them (retinaface).
- `pose` = None: DeepFace reports no head angles. The gate derives roll from the eye line and
  `yaw_ratio` from the nose, so `yaw` and `pitch` stay `no_pose`.
- No geometry for `detector_backend="skip"` (the region is the whole frame) or for the
  `enforce_detection=False` fallback (confidence 0 and no eye).
- Coordinates are in the grid of the array passed to DeepFace, the Pillow decode without EXIF
  transpose that the gate also reads.

**Scope.** `verify`, `identify`, the dedup candidate source, `consolidate` and `purge` ignore
the verdict.

### 6.8 Risk profiles

`biometric/risk_profiles.py` holds named profiles under `BIOMETRIC["RISK_PROFILES"]` (module
configuration key `risk_profiles`, default `{}`). A profile is a partial override that can only
tighten the base: `BIOMETRIC["FUSION"]` for `fuse()` and `BIOMETRIC["MODALITIES"][m]["threshold"]`
for `verify()` and for the per-leg normalisation in `fuse()`.

```python
"RISK_PROFILES": {
    "high_risk": {
        "thresholds": {"accept": 0.9},             # accept / review, may only rise
        "floors": {"fingerprint": 60},             # may only rise; a new floor is allowed
        "floor_decision": "reject",                # "review" -> "reject" only
        "required": ["fingerprint"],               # union with the caller's set
        "modality_thresholds": {"face": 0.8},      # raw provider scale, configured modalities only
    },
},
```

**Keys.** `thresholds`, `floors`, `floor_decision`, `required`, `modality_thresholds`. `weights`
is refused: weights have no strictness order, and moving weight onto one leg can accept a score
vector the base reviews. A profile that needs a stronger leg uses `floors`, `required` or
`modality_thresholds`.

**Validation** (`profile_errors`, `validate_profiles`) refuses, with a message naming the profile
and the key:
- a name that is not a string, is blank or is longer than 64 characters;
- overrides that are not a non-empty dict; an unknown key; `weights`;
- a threshold, floor or modality threshold that is not a finite number (bool excluded), a
  negative threshold or floor, or a value below the base;
- a merged `review` above the merged `accept`;
- a `floor_decision` outside `review`/`reject`, or `review` when the base is `reject`;
- `required` that is not a list of modality names;
- a `modality_thresholds` entry for a modality with no configured numeric threshold.

`BiometricConfig.ready()` logs every error (`biometric.apps`, level ERROR) and never raises, so a
bad module configuration row does not stop the backend or `manage.py migrate`. `resolve()` validates
the named profile again on each call against the configured base and raises `RiskProfileError`
(an `ImproperlyConfigured`), which covers configuration changed at runtime.

**Merge.** `tighten()` merges per key onto the rules the caller resolved: `max` for `accept`,
`review`, each floor and each modality threshold; the stricter `floor_decision`; the union of
`required`. A key the profile omits keeps the base value, and explicit `thresholds=`/`floors=`/
`floor_decision=`/`required=` arguments to `fuse()` are clamped the same way, so a profile never
loosens them. In `fuse()` a leg with a raised modality threshold is normalised to the lower of
`score / base_threshold` and `score / profile_threshold`, so a negative leg score is never lifted.

**Unknown name.** `fuse()` and `verify()` raise `UnknownRiskProfileError` (a `ValueError`)
synchronously: `fuse()` before any scoring, `verify()` before extraction and before any row is
written. Over GraphQL it is an error on `verifyBiometric`, never `verified: false`. Choosing a
profile needs no right beyond `gql_biometric_verify_perms` (174002).

**Recorded.** `verify()` writes the profile name to `BiometricVerification.risk_profile` and the
effective threshold, `max(base, profile)`, to `threshold`. A profile that does not name the
verified modality leaves the threshold unchanged and still records its name. `fuse()` writes
nothing; its `Decision.risk_profile` carries the name.

**Where each key applies.** `verify()` and `verifyBiometric` decide one modality: they apply
`modality_thresholds` only. `verify_multimodal()` and `verifyBiometricMultimodal` (§6.11) apply
every key: each leg's `verify()` takes the raised modality threshold, and the fused decision takes
`thresholds`, `floors`, `floor_decision`, `required` and the normalisation by the raised modality
thresholds.

**Default.** With `risk_profile` `None` or `""`, `fuse()` and `verify()` never call into
`risk_profiles` and apply the base rules unchanged. A profile changes only the fusion rules and
the modality thresholds listed above; no other threshold in the module reads it.

### 6.9 Impersonation probe

`biometric/impersonation.py` runs an optional 1:N search inside `verify()`: the probe already
extracted for the 1:1 comparison (server path), or the vector / template the device supplies
(device path, opt-in), is ranked against the whole gallery of its modality. A foreign subject at or above the probe threshold is reported as a possible
impersonation. The probe never changes `score`, `threshold` or `verified`.

```python
"IMPERSONATION_PROBE": {
    "enabled": False,          # master switch; False leaves verify() unchanged
    "modalities": ["face"],    # modalities whose verify() runs the probe
    "top_k": 5,                # foreign subjects kept
    "thresholds": {},          # {modality: similarity}; see the fallback chain below
    "margin": None,            # None, or a float narrowing the suspect rule
    "device_path": False,      # True: rank a device-supplied vector / template on the device path
},
```

Module configuration key `impersonation_probe`. `probe_settings()` merges the configured dict
over these defaults at read time, so `{"enabled": True}` alone uses the other defaults, and
`{}` or `None` leaves the probe off.

**When it runs.** With `enabled` true and the modality listed:

- Server path (a `sample` is given): always, on the extracted probe.
- Device path (`device_score` given): the server holds no sample, so the probe ranks what the
  device extracted, passed as `verify(device_template=Extracted(...))`: `vector` for an embedding
  modality, `template` for a template modality. It runs only when `device_path` is true. The
  first applicable reason, in this order, is recorded when it does not run:
  - `provider_matches_on_device`: the modality's provider is `DeviceReportedMatcher`, whose
    `match()` raises, so the gallery cannot be ranked on the server whatever the device sends;
  - `no_device_template`: the device supplied no vector (embedding kind) or template (template
    kind);
  - `device_path_disabled`: `device_path` is false.

  The device's vector or template must come from the gallery's provider and model; nothing checks
  it, as for a device template at `enrol()`. A device that misreports its score can also send a
  template that matches nobody: the device-path probe detects only what the device honestly sends.
  `device_template` with a `sample` raises `ValueError`.

No `verifyBiometric` argument turns the probe on or off.

**Threshold.** The first value set among `IMPERSONATION_PROBE["thresholds"][m]`,
`DEDUP_THRESHOLD[m]`, `MODALITIES[m]["threshold"]`, then the provider's `default_threshold`. It
is on the scale `identify()` returns (similarity for embeddings, the matcher's score for
templates). A risk profile (§6.8) never changes it.

**Search and exclusion.** Inside a savepoint (`transaction.atomic()`), the probe counts `n`, the
claimed subject's active templates on the gallery key (modality, provider, model name, kind),
then calls `identify(modality, vector=..., template=..., top_k=top_k + n)`. It never passes
`sample` (no second extraction) and never passes `actor`. The claimed subject is dropped in
Python by its `(subject_model, subject_id)` pair; `claimed_score` is the best of its dropped
rows, or `None` when none surfaced. Reading `n` extra rows keeps `top_k` foreign rows when all
of the claimed subject's templates rank first. A foreign subject with the same `subject_id` under
another `subject_model` is still a candidate. `identify()` and its helpers are unchanged, so the
numpy and pgvector paths both apply.

**Candidates and suspicion.** Foreign rows are collapsed per subject (best template), filtered
at `score >= threshold` and truncated to `top_k`. Each candidate is
`{subject_model, subject_id, template_id, score, suspect}` with
`suspect = margin is None or claimed_score is None or score >= claimed_score - margin`.
`suspected` is true when any candidate is a suspect; `best_match` is the first suspect. With
`margin` `None`, any foreign match at or above the threshold is a suspect, whatever the claimed
subject's score.

**Failure.** Any exception inside the probe is recorded as status `failed`, `suspected` false,
and `error` = `"<ExceptionClass>: impersonation probe failed"`. The exception message is never
stored, returned or logged: it can quote a stored value or the probe vector (psycopg2
interpolates parameters into its error text). A gallery row that does not decrypt records
`TemplateKeyError: impersonation probe failed` (§3.3).
`biometric.impersonation` logs a WARNING with the modality, that error and the stack frames only. `verify()` still returns and
writes its row; the savepoint keeps a database error from aborting the caller's transaction. A
modality on `DeviceReportedMatcher` listed here records `failed` on every server-path verify,
because its `match()` raises.

**Recorded.** Seven columns on `BiometricVerification`:

| Column | Content |
|---|---|
| `impersonation_status` | `""` not run, `"ok"`, `"failed"` |
| `impersonation_suspected` | bool, default false |
| `impersonation_subject_model` / `impersonation_subject_id` | the best suspect, `""` otherwise |
| `impersonation_score` | the best suspect's score, null otherwise |
| `impersonation_evidence` | JSON: `threshold`, `margin`, `top_k`, `claimed_score`, `candidates`, `error`, `latency_ms` |
| `impersonation_skip_reason` | `""`, or the reason an enabled probe did not run on the device path (migration 0003) |

`VerificationResult.impersonation_skip_reason` carries the same reason as the column.
`VerificationResult.impersonation` carries the `ImpersonationProbe`, or `None` when the probe
did not run.

**Signal.** On a suspicion, `verify()` logs a warning naming the verification id and modality
only, then registers `transaction.on_commit` to emit the service signal
`biometric.impersonation_suspected` (registered by `biometric/signals.py`). Subscribers bind
with `ServiceSignalBindType.AFTER` and read the payload from `kwargs["result"]`:

```python
{"verification_id", "subject_model", "subject_id", "modality",
 "matched_subject_model", "matched_subject_id", "matched_template_id", "matched_score",
 "claimed_score", "threshold", "margin", "actor", "device_id", "context"}
```

Core service signals use `Signal.send`, so a raising receiver would propagate; the emitter logs
and swallows it. The payload links two subjects: a subscriber applies its own access rights.

**GraphQL.** `verifyBiometric` and `biometricVerifications` rows expose `impersonation`
(`BiometricImpersonationProbeType`: `status`, `suspected`, `threshold`, `margin`, `topK`, `claimedScore`,
`matchedSubjectModel`, `matchedSubjectId`, `matchedScore`, `candidates`, `error`, `latencyMs`),
null when the probe did not run. `matchedSubjectModel`, `matchedSubjectId` and `candidates` are
filled only for a caller holding `gql_biometric_identify_perms` (174003); with 174002 or 174004
alone the caller gets the status, the verdict and the scores.
Both also expose `impersonationSkipReason: String`. `verifyBiometric` takes `deviceVector: [Float!]`
and `deviceTemplate: String` (base64) for the device-path probe; `BiometricVerifyLegInput` of
`verifyBiometricMultimodal` takes the same two fields.

**Deduplication.** `biometric` never reads deduplication tables (§2). A CONFIRMED merge removes
the echo because `consolidate()` moves or supersedes the retired subject's active templates. An
OPEN `DuplicateCandidate` pair is not suppressed: verifying either subject raises a suspicion
naming the other until the pair is resolved. The signal payload names both subjects, so a
subscriber can reconcile it.

**Cost.** On the numpy path each probed verify decrypts and ranks every active template of the
modality; `latency_ms` is recorded in the evidence. Large galleries use `VECTOR_INDEX="pgvector"`.

### 6.10 Audit chain and alerts

`biometric/audit_chain.py` appends hash-chained audit events. `biometric/audit_rules.py` raises
alerts from them. The feature is off by default: while `BIOMETRIC["AUDIT"]["enabled"]` is not the
boolean `True`, no event or alert row is written, no advisory lock is taken, no existing path
gains a transaction (`audited_block()` returns `contextlib.nullcontext()`), and every return
value is unchanged.

```python
"AUDIT": {
    "enabled": False,   # master switch; only the boolean True turns it on
    "rules": {},        # {rule kind: {param: value}}; overrides DEFAULT_RULES key by key
},
```

Module configuration key `audit`. A settings override replaces the whole dict, so every reader
uses `.get()`. Only `enabled` and `rules` are valid top-level keys.

**Models** (migration generated with `makemigrations`; no data migration):

| Model | Fields |
|---|---|
| `BiometricAuditEvent` (`biometric_audit_event`) | `id` UUID pk; `sequence` bigint UNIQUE; `action` char(48); `actor` char(64); `subject_model` char(64); `subject_id` char(64) indexed; `modality` char(16); `payload` JSON; `created_at` indexed; `prev_hash` char(64) UNIQUE; `hash` char(64). Indexes `biometric_audit_subject_idx` (action, subject_model, subject_id, created_at) and `biometric_audit_actor_idx` (actor, action, created_at). |
| `BiometricAlert` (`biometric_alert`) | `id` UUID pk; `rule_kind` char(48); `severity` LOW / MEDIUM / HIGH; `title` char(200); `detail` JSON; `dedupe_key` char(200); `subject_model`; `subject_id` indexed; `trigger_event` FK PROTECT; `state` NEW / ACKNOWLEDGED / RESOLVED; `occurrences`; `triggered_at`; `last_seen_at`; `acknowledged_by` / `_at`; `resolved_by` / `_at`; `resolution_note`. Partial UNIQUE `biometric_alert_one_open_per_key` on (rule_kind, dedupe_key) where state is NEW or ACKNOWLEDGED. |

Events are append-only. The manager's `update()` and `delete()` raise `PermissionError`, and so do
`save()` on an existing row and `delete()` on an instance. The actor is the username string, as
on `BiometricVerification`.

**Actions** (`audit_chain.ACTIONS`):

| Action | Recorded by | Subject | Payload |
|---|---|---|---|
| `template.enrol` | `enrol()` | enrolled subject | `template_id`, `position`, `provider`, `model_name`, `kind`, `quality`, `encrypted`, `device_template`, `superseded`, `quality_status`, `quality_reasons` |
| `template.enrol_refused` | `enrol()`, enforce mode, `REFUSED` verdict | subject of the refused sample | `position`, `provider`, `model_name`, `kind`, `quality`, `device_template`, `quality_status`, `quality_mode`, `quality_reasons`, `quality_measures` |
| `verify` | `verify()` | claimed subject | `verification_id`, `verified`, `score`, `threshold`, `origin`, `fallback`, `device_id`, `position`, `risk_profile`, `impersonation_status`, `impersonation_suspected`, `impersonation_skip_reason`, `template_skip_reason` |
| `verify.multimodal` | `verify_multimodal()`, after the legs | claimed subject | `decision_id`, `outcome`, `score`, `reasons`, `risk_profile`, `modalities`, `verification_ids`, `fallback`, `device_id` |
| `impersonation.suspected` | `verify()` through `record_impersonation_suspected()` | claimed subject | `verification_id`, `matched_subject_model`, `matched_subject_id`, `matched_template_id`, `matched_score`, `claimed_score`, `threshold`, `margin` |
| `identify` | `identify(actor=...)` only | none | `top_k`, `scope`, `exclude_subject`, `probe` (`sample` / `vector` / `template`), `matches` [{`subject_model`, `subject_id`, `template_id`, `score`}] |
| `template.consolidate` | `consolidate()`, only when templates moved | kept subject | `retired_id`, `counts`, `template_ids` |
| `template.purge` | `purge()`, one per tombstone | erased subject | `erasure_id`, `reason`, `erased` |
| `template.read` | `templates_of()` | read subject | `purpose`, `template_ids`, `modality` |
| `template.list` | `biometricTemplates` resolver | listed subject | `template_ids` |
| `alert.acknowledge` / `alert.resolve` | triage services | the alert's subject | `alert_id`, `rule_kind`, `note` |

The caller's `context` is never copied into a payload. `sanitize_payload()` refuses the keys
`sample`, `vector`, `template`, `template_iso`, `embedding`, `image`, `frame` and `probe_vector` at
any depth, lists holding more than 16 numbers, and NaN or infinity (`ValueError`). It also refuses
binary values (`TypeError`). It converts UUIDs, dates and NumPy scalars, writes negative zero as
`0.0` and integral floats of magnitude 1e16 or more as integers (the values Postgres `jsonb` returns),
and returns the JSON round trip of the result, so the stored value and the hashed value are the same. A refused payload
writes nothing and consumes no sequence.

**Where events are written.** Each producer records its event as the last write of an
`audited_block()`, so the event commits or rolls back with the business rows:

- `enrol()`: the supersede loop, the insert and the event. The quality gate runs before the block.
  A sample refused in enforce mode writes no template and records `template.enrol_refused` in its
  own block; `QualityRefusedError` is raised after that block. A caller transaction that rolls
  back on the error discards the event. openIMIS runs GraphQL mutations outside a transaction
  unless `ATOMIC_MUTATIONS` is set, so a refused `enrolBiometric` keeps its event.
- `verify()`: the row and the `verify` event, then the `impersonation.suspected` event when the
  probe suspects someone. The risk profile and the probe (§6.9) run before the block, so the lock is
  never held during the probe's gallery search. The impersonation event is recorded here, inside
  `verify()`. It is never recorded from the `biometric.impersonation_suspected` signal, which still
  fires after commit as in §6.9.
- `consolidate()`: inside its existing transaction.
- `templates_of()`: the access log and the event.
- `_erase()` (called by `purge()`): the delete, the tombstones, then one event per tombstone.

The impersonation probe and `BiometricCandidateSource.scan` call `identify()` without an actor and
record nothing.

**Hash and lock.** `canonical_event(row)` is `json.dumps` of `{id, sequence, action, actor,
subject_model, subject_id, modality, payload, created_at}` with `sort_keys=True`, separators `,`
and `:`, and `allow_nan=False`. `created_at` is ISO with microseconds; an aware value is converted
to naive UTC first. `hash = sha256(prev_hash || canonical_event(row))`, and
`GENESIS_HASH = "0" * 64` is the `prev_hash` of sequence 1. `record_event()` takes
`pg_advisory_xact_lock(0x42494F4155444954)` (Postgres only), reads the head, and inserts
`sequence = head + 1`. The lock is held until the outermost transaction commits. The unique
`sequence` and `prev_hash` turn an append that escaped the lock into an `IntegrityError`, not a fork.

**Verifier.** `verify_chain(batch_size=1000)` walks the rows by `sequence` from `GENESIS_HASH` and
returns a `ChainReport(checked, head_sequence, head_hash, divergence)`. The divergence is the first
of:

- `missing_event`: a sequence gap, including a deleted first row.
- `broken_link`: `prev_hash` differs from the previous row's `hash`.
- `altered_row`: the recomputed hash differs from the stored one.

**Command.** `manage.py biometric_audit_verify [--batch-size N] [--expected-head HASH]
[--expected-sequence N]` prints `biometric audit chain intact over {n} event(s); head sequence {s}
hash {h}`. It raises `CommandError` on a divergence
(`biometric audit chain diverges after {checked} event(s): {kind} at sequence {seq}: {detail}`) and
when a recorded head no longer holds (`audit_chain.check_anchor()`). Each run stores its walk as the
audit chain status (§6.12). The expected values are a head
printed by an earlier run; the chain may have grown since. The event at `--expected-sequence` must
still exist and hold `--expected-head`. Given alone, `--expected-sequence` must not exceed the
current head sequence and `--expected-head` must be the hash of some event. Sequence 0 with
`GENESIS_HASH`, the head of an empty chain, holds on any chain.

**What the chain proves, and what it does not.**

- It proves that no stored row was altered, deleted or relinked in the middle of the chain.
- Deleting the newest rows leaves a coherent chain. Only a head recorded outside this database and
  passed back as `--expected-sequence` / `--expected-head` reveals it. It covers the events up to
  that head; events appended later are covered once a newer head is recorded.
- Someone holding both the database and the application can recompute a coherent chain.
- `created_at` is asserted by this server. No timestamp authority signs it.
- Changing `TIME_ZONE` after events exist changes every recomputed hash (openIMIS runs
  `USE_TZ = False`).
- Events have no retention. They hold `subject_id` and `actor`, but no biometric material.

**Rules.** `evaluate_event(event_id)` runs from `transaction.on_commit` after each event. It skips
missing events and every `alert.*` action. Each enabled rule runs in its own savepoint inside a
try/except that logs and continues, so a failing rule never fails the committed operation or the
other rules.

| Kind | Defaults | Fires when | Dedupe key |
|---|---|---|---|
| `FAILED_VERIFICATIONS` | `enabled` True, `severity` MEDIUM, `threshold` 3, `window_minutes` 60, `per_modality` True | a `verify` event with `verified` false brings the subject's failed verifies in `[created_at - window, created_at]` (same modality when `per_modality`) to `threshold` or more | `failed_verifications:{subject_model}:{subject_id}[:{modality}]` |
| `IMPERSONATION_SUSPECTED` | `enabled` True, `severity` HIGH | every `impersonation.suspected` event; the alert's subject is the claimed subject | `impersonation:{subject_model}:{subject_id}:{matched_subject_model}:{matched_subject_id}` |
| `ACCESS_BURST` | `enabled` True, `severity` HIGH, `max_events` 200, `window_minutes` 60, `actions` [`template.read`, `template.list`, `identify`] | one non-blank actor records `max_events` or more watched actions in the window | `access_burst:{actor}` |

Alert details:

- `FAILED_VERIFICATIONS`: `attempts`, `threshold`, `window_minutes`, `modality`, `actors`,
  `device_ids`.
- `IMPERSONATION_SUSPECTED`: `verification_id`, `modality`, `matched_subject_model`,
  `matched_subject_id`, `matched_template_id`, `matched_score`, `claimed_score`, `threshold`,
  `margin`.
- `ACCESS_BURST`: `actor`, `events`, `max_events`, `window_minutes`, `actions`,
  `distinct_subjects`.

`raise_alert()` locks the open alert (NEW or ACKNOWLEDGED) with the same `(rule_kind, dedupe_key)`
and bumps `occurrences`, `last_seen_at`, `detail` and `trigger_event`. Otherwise it creates the
alert; when a concurrent creator wins the partial unique constraint, it re-reads and bumps. After
RESOLVED, the next occurrence opens a new alert.

**Configuration checks.** `validate_audit_config(cfg)` returns a list of messages and never raises.
It reports:

- an unknown top-level key or rule kind;
- an unknown parameter;
- a non-boolean `enabled` or `per_modality`;
- a severity outside LOW / MEDIUM / HIGH;
- a `threshold`, `window_minutes` or `max_events` that is not a positive, non-boolean integer;
- `actions` that is empty, or that holds an unknown action or an `alert.*` action.

`BiometricConfig.ready()` logs each message as `BIOMETRIC['AUDIT']: …` and never raises.
`rule_params(kind)` merges the defaults with the override and raises `ImproperlyConfigured` on an
invalid result. `evaluate_event()` logs that error and skips the rule.

**Triage.** `acknowledge_alert(alert_id, *, actor)` moves an alert from NEW to ACKNOWLEDGED.
`resolve_alert(alert_id, *, actor, note="")` moves it from NEW or ACKNOWLEDGED to RESOLVED. Each
locks the alert, applies `BiometricAlert.acknowledge()` / `.resolve()` (`ValidationError` on any
other transition) and records `alert.acknowledge` / `alert.resolve`.

**Rights.**

- `gql_biometric_audit_perms` (`174005`) reads events and alerts.
- `gql_biometric_alert_perms` (`174006`) acknowledges and resolves alerts.
- `gql_biometric_config_perms` (`174007`) reads the decision criteria and the retention policy
  (§6.12).
- `gql_biometric_audit_verify_perms` (`174008`), together with 174005, runs the chain
  verification (§6.12).

Settings keys are `GQL_BIOMETRIC_AUDIT_PERMS`, `GQL_BIOMETRIC_ALERT_PERMS`,
`GQL_BIOMETRIC_CONFIG_PERMS` and `GQL_BIOMETRIC_AUDIT_VERIFY_PERMS`. These modules
grant them to no role.

**GraphQL** (relay connections, `ExtendedConnection`, page size capped by
`RELAY_CONNECTION_MAX_LIMIT`):

- `biometricAuditEvents`: `BiometricAuditEventGQLType` nodes, newest sequence first.
  - Node fields: `id`, `sequence`, `action`, `actor`, `subjectModel`, `subjectId`, `modality`,
    `payload`, `createdAt`, `prevHash`, `hash`.
  - Arguments: `sequence`, `sequence_Lt`, `sequence_Gt`, `action`, `action_Startswith`, `actor`,
    `subjectModel`, `subjectId`, `modality`, `createdAt_Gte`, `createdAt_Lte`, `orderBy`, `first`,
    `after`, `before`, `last`, `offset`.
  - Right: 174005.
- `biometricAlerts`: `BiometricAlertGQLType` nodes ordered by `-triggeredAt`, `-id`.
  - Node fields: `id`, `ruleKind`, `severity`, `state`, `title`, `detail`, `subjectModel`,
    `subjectId`, `occurrences`, `triggeredAt`, `lastSeenAt`, `triggerEventId`, `acknowledgedBy`,
    `acknowledgedAt`, `resolvedBy`, `resolvedAt`, `resolutionNote`. `dedupeKey` is not exposed.
  - Arguments: `state`, `severity`, `ruleKind`, `open` (true: NEW or ACKNOWLEDGED only; false:
    RESOLVED only), `subjectModel`, `subjectId`, `triggeredAt_Gte`, `triggeredAt_Lte`, `orderBy`,
    `first`, `after`, `before`, `last`, `offset`.
  - Right: 174005.
- `acknowledgeBiometricAlert(id: String!)` and `resolveBiometricAlert(id: String!, note: String)`
  take the alert's raw UUID and return the alert. An invalid transition is a GraphQL error.
  Right: 174006.
- **Identity stripping.** Without `gql_biometric_identify_perms` (174003):
  - an `identify` event's `matches` lose `subject_model`, `subject_id` and `template_id`, keeping
    `score`;
  - an `impersonation.suspected` event's payload and an `IMPERSONATION_SUSPECTED` alert's detail
    lose `matched_subject_model`, `matched_subject_id` and `matched_template_id`.

  The stored rows are unchanged.
- **Relay `node`.** The assembled openIMIS schema declares a root `node` field that resolves a
  global id through the type's `get_queryset`. `BiometricAuditEventGQLType`,
  `BiometricAlertGQLType` and `BiometricErasureGQLType` require 174005 there too, so `node`
  returns a `PermissionDenied` error without it.
- `identifyBiometric` passes `actor=user.username` to `identify()`, and `biometricTemplates`
  records `template.list` when audit is on. Their arguments and results are unchanged.

### 6.11 Multimodal verification

`services.verify_multimodal()` verifies one subject on several modalities and fuses the scores.

```python
verify_multimodal(subject_model=None, subject_id=None, legs=None, *, fallback=False, context=None,
                  device_id="", actor, risk_profile=None) -> MultimodalVerification
# legs: [{"modality", "sample" | "device_score", "position"?, "device_template"?}]
# MultimodalVerification(decision: Decision, legs: list[VerificationResult])   # legs in input order
```

- Each leg runs `verify()` with the shared `fallback`, `context`, `device_id`, `actor` and
  `risk_profile`, and records its own `BiometricVerification` row and audit event.
- The fused decision is `fuse({leg modality: leg score}, risk_profile=risk_profile)`: the
  configured `BIOMETRIC["FUSION"]` rules tightened by the profile (§6.8). A leg with no score (no
  active template) counts as missing, so a `required` modality caps the decision at review.
- The caller passes no weights, thresholds, floors, floor decision or required set: without a
  profile, explicit `fuse()` arguments are taken as given and could loosen the configured base.
- Checked before any leg runs, so none of these write a row:
  - `ValueError` for a leg list that is empty, has a non-dict leg, an unknown key, a missing
    modality, a modality twice, a leg without exactly one of `sample` and `device_score`, or a
    `device_template` with a `sample`;
  - `KeyError` for a modality with no registered provider;
  - `UnknownRiskProfileError` / `RiskProfileError` for the profile.
- A leg that fails later (for example no face in its sample) raises and keeps the rows of the legs
  before it. The legs do not share a transaction: with audit on, the chain lock of one leg is never
  held during the next leg's impersonation probe.
- The fused decision is stored after the last leg as a `BiometricMultimodalDecision` row
  (`biometric_multimodal_decision`, migration 0005): `subject_model`, `subject_id`, `outcome`,
  `score` (null when no weighted leg scored), `reasons`, `risk_profile`, `modalities` and
  `verification_ids` (the legs' `BiometricVerification` ids, in leg order), `fallback`, `device_id`,
  `actor`, `created_at`. It is written whether audit is on or off. A leg list or profile refused up
  front, or a leg that raises, stores no decision. `verify()` returns its row id as
  `VerificationResult.verification_id`; `MultimodalVerification.decision_id` is the decision's id.
- With audit on (§6.10), the row and a `verify.multimodal` event share one `audited_block()`, after
  the legs' own `verify` events. The event's subject is the claimed subject and its modality is
  empty. Payload: `decision_id`, `outcome`, `score`, `reasons`, `risk_profile`, `modalities`,
  `verification_ids`, `fallback`, `device_id`; no sample, vector or template. It names no other
  subject: a leg's impersonation match stays on that leg's row and its `impersonation.suspected`
  event, stripped as §6.10 describes, so there is nothing to strip from the decision.

GraphQL mutation `verifyBiometricMultimodal`, right `gql_biometric_verify_perms` (174002):

- Arguments: `subjectId: String!`, `subjectModel: String`, `legs: [BiometricVerifyLegInput!]!`,
  `riskProfile: String`, `fallback: Boolean`, `deviceId: String`, `context: JSONString`.
- `BiometricVerifyLegInput`: `modality: String!`, `sample: String` (base64), `position: String`,
  `deviceScore: Float`, `deviceVector: [Float!]`, `deviceTemplate: String` (base64).
- Result `BiometricMultimodalVerifyResultType`: `decisionId` (UUID of the stored decision), `outcome`
  (accept | review | reject), `score`, `reasons`, `riskProfile`, `legs: [BiometricVerifyResultType!]!`.
  `BiometricVerifyResultType.verificationId` is the UUID of the leg's (or `verifyBiometric`'s)
  `BiometricVerification` row.
- An invalid leg list or profile is a GraphQL error and no verification is recorded.

Query `biometricMultimodalDecisions`, right `gql_biometric_read_perms` (174004, the right of
`biometricVerifications`), relay connection (`ExtendedConnection`, page size capped by
`RELAY_CONNECTION_MAX_LIMIT`), newest `createdAt` first:

- Node `BiometricMultimodalDecisionGQLType`: `id`, `subjectModel`, `subjectId`, `outcome`, `score`,
  `reasons`, `riskProfile`, `modalities`, `verificationIds`, `fallback`, `deviceId`, `actor`,
  `createdAt`.
- Arguments: `subjectModel`, `subjectId`, `outcome`, `riskProfile`, `createdAt_Gte`, `createdAt_Lte`,
  `orderBy`, `first`, `after`, `before`, `last`, `offset`.
- The root `node` lookup of this type needs 174004 as well.

### 6.12 Admin queries

Read-only views of the configuration and the retention and audit state. No field exposes
`TEMPLATE_KEY`, provider settings, the dedupe key of an alert, or any biometric material.

| Field | Kind | Right | Returns |
|---|---|---|---|
| `biometricDecisionCriteria` | query | 174007 | `BiometricDecisionCriteriaType` |
| `biometricRetentionPolicy` | query | 174007 | `BiometricRetentionPolicyType`, null without a policy row |
| `biometricErasures` | relay connection | 174005 | `BiometricErasureGQLType` nodes |
| `biometricErasureFilterValues` | query | 174005 | `BiometricErasureFilterValuesType` |
| `biometricAuditChainStatus` | query | 174005 | `BiometricAuditChainCheckType`, null before the first check |
| `verifyBiometricAuditChain` | mutation | 174008 and 174005 | `BiometricAuditChainCheckType` |

**Decision criteria** (`risk_profiles.decision_criteria()`), built from `BIOMETRIC["FUSION"]`,
the numeric `MODALITIES[m]["threshold"]` values and `BIOMETRIC["RISK_PROFILES"]` only:

- `base: BiometricFusionRulesType`: `acceptThreshold`, `reviewThreshold`, `floors`,
  `floorDecision`, `required`, `modalityThresholds`, `weights`. The three lists hold
  `BiometricModalityValueType { modality, value }`. A modality with no configured threshold is
  absent from `modalityThresholds`; `verify()` uses its provider's default threshold.
- `profiles: [BiometricRiskProfileType!]!`, sorted by name: `name`, `valid`, `errors` (the §6.8
  validation messages), `overrides: BiometricRiskProfileOverridesType` (`acceptThreshold`,
  `reviewThreshold`, `floors`, `floorDecision`, `required`, `modalityThresholds`, as declared) and
  `effective: BiometricFusionRulesType` (the profile merged onto the base, null when invalid).
- In `overrides`, only the five profile keys are read. A value of the wrong type is null (numbers,
  `floorDecision`) or left out (`required` entries that are not strings). `weights` and unknown
  keys are never shown; `errors` names them.

**Retention policy** (`services.current_retention_policy()`, the row `purge()` applies):
`templateRetentionDays`, `purgeEnabled`, `activeTemplateRetentionDays`, `purgeActiveEnabled`.

**Erasures.** Nodes `id`, `subjectModel`, `subjectId`, `modalities: [String]`, `erased`
(JSON counts per modality), `reason`, `erasedBy`, `erasedAt`, newest `erasedAt` first. Arguments:
`subjectModel`, `subjectId`, `reason`, `erasedBy`, `erasedAt_Gte`, `erasedAt_Lte`, `orderBy`,
`first`, `after`, `before`, `last`, `offset`; page size capped by `RELAY_CONNECTION_MAX_LIMIT`.

**Erasure filter values.** `biometricErasureFilterValues` (no arguments) returns
`BiometricErasureFilterValuesType { erasedBy: [String!]!, subjectModel: [String!]! }`: the distinct
`erased_by` and `subject_model` values of `BiometricErasure`, each sorted ascending and cut to the
first `schema.ERASURE_FILTER_VALUES_LIMIT` (200) in that order. Distinct, sort and cut run in SQL.

**Chain status.** `audit_chain.record_chain_check(actor=...)` runs `verify_chain()` and stores a
`BiometricAuditChainCheck` row (`biometric_audit_chain_check`, migration 0003): `checked_at`,
`checked_by`, `ok`, `checked`, `head_sequence`, `head_hash`, `divergence_kind`,
`divergence_sequence`, `divergence_detail`. The row is not an audit event, so the head it records
stays the head it verified. `verifyBiometricAuditChain` (no arguments) calls it with the caller's
username. `biometricAuditChainStatus` returns the latest row
(`audit_chain.latest_chain_check()`). Type fields: `id`, `ok`, `checkedAt`, `checkedBy`, `checked`,
`headSequence`, `headHash`, `divergenceKind` (`missing_event` | `broken_link` | `altered_row`,
empty when intact), `divergenceSequence` (first divergent sequence, null when intact),
`divergenceDetail`.

The `biometric_audit_verify` command walks the chain once and stores the walk through
`audit_chain.store_chain_check()`, the service `record_chain_check()` calls, with `checked_by`
`"biometric_audit_verify"`, before it prints or raises. A scheduled run therefore updates
`biometricAuditChainStatus`, a diverging run included (the row is stored, then the command exits
with its `CommandError`). The row records the walk only: with `--expected-sequence` /
`--expected-head`, an intact walk whose recorded head no longer holds stores `ok` true and still
fails the command, so a truncated tail shows in the exit code, not in the status.

### 6.13 Preprocessing tag

Two vectors are comparable only when the provider turned both samples into model input the same
way. Each provider declares that way as `ModalityProvider.preprocessing` (a string, `""` for
none), and every template records the value it was extracted under.

**DeepFace colour order.** `DeepFace.represent()` takes a numpy array in BGR channel order, the
OpenCV convention: the `represent()` docstring says so for `img_path`, `commons/image_utils.load_image`
returns a numpy input unchanged as the BGR image, and `modules/representation.py` reverses the
detected face from RGB to BGR before the model (checked on deepface 0.0.101). `_to_numpy()` decodes
the sample with Pillow, without EXIF transpose, converts it to RGB and reverses the channels into a
contiguous BGR array. `DeepFaceProvider.preprocessing` is `"pillow_bgr"`. Face geometry (§6.7) is
unaffected: reversing channels moves no pixel.

**Recorded.** `enrol()` writes the provider's tag to `BiometricTemplate.metadata["preprocessing"]`
when the tag is not empty, for a server extraction and for a device template alike (the device
template is taken to come from the gallery's provider, as its provider and model name are). The
key is reserved: a caller's `metadata["preprocessing"]`, or one in the device's
`Extracted.metadata`, is dropped. A row without the key carries `""`.

**Rule.** A template is compared with a provider's output only when its tag equals the provider's
current tag (`services.comparable_preprocessing`). Otherwise it is skipped with the reason
`preprocessing_mismatch`:

| Where | Skip | Reason recorded |
|---|---|---|
| `verify()`, server path | the subject's template is not compared | `BiometricVerification.template_skip_reason` (migration 0004), `VerificationResult.template_skip_reason`, the `verify` audit event's `template_skip_reason`, a WARNING from `biometric.services` with the count |
| `identify()`, numpy and template paths | the row leaves the gallery before ranking, so it never takes a `top_k` place | INFO from `biometric.services` with the count |
| `identify()`, pgvector path | `COALESCE(bt.metadata->>'preprocessing', '') = <provider tag>` in the query, before `LIMIT` | none (the query returns no count) |
| impersonation probe (§6.9) | through `identify()` | as `identify()` |
| `BiometricCandidateSource.scan` | a probe row under another tag is not scanned; its gallery goes through `identify()` | INFO from `biometric.dedup_source` with the count |

`template_skip_reason` is `""` when nothing was skipped, and on the device path, which compares no
stored template on the server. When every template of the subject is skipped, `score` is null and
`verified` false, as for a subject with no template.

GraphQL: `templateSkipReason: String` on `BiometricVerifyResultType` (so on `verifyBiometric` and on
each leg of `verifyBiometricMultimodal`) and on `biometricVerifications` rows.

**Migration of a gallery.** No gallery extracted before the tag exists in any deployment. A template
under another tag stays stored and inert: re-enrolling the subject supersedes it, since the unique
active key (subject, modality, position, provider, model name) does not include the tag. A provider
whose preprocessing changes changes its tag, and every template recorded under the old tag stops
matching until it is re-enrolled.

### 6.14 Verification records

`biometricVerifications(subjectId!, subjectModel)` keeps its shape: a list for one subject. The
records across subjects are read through a relay connection, right `gql_biometric_read_perms`
(174004), newest `createdAt` first, page size capped by `RELAY_CONNECTION_MAX_LIMIT`:

- `biometricVerificationRecords`, node `BiometricVerificationGQLType`: `id`, `subjectModel`,
  `subjectId`, `modality`, `score`, `threshold`, `verified`, `origin`, `fallback`, `deviceId`,
  `actor`, `createdAt`, `riskProfile`, `impersonation` (`BiometricImpersonationProbeType`, with
  `topK`; §6.9), `impersonationSkipReason`, `templateSkipReason` (§6.13).
- Arguments: `subjectModel`, `subjectId`, `modality`, `suspected` (true: rows whose probe suspected
  someone; false: the others), `createdAt_Gte`, `createdAt_Lte`, `orderBy`, `first`, `after`,
  `before`, `last`, `offset`.
- `impersonation` strips `matchedSubjectModel`, `matchedSubjectId` and `candidates` without
  `gql_biometric_identify_perms` (174003), as on `biometricVerifications`. The caller's `context`,
  the raw `impersonation_evidence` and the `impersonation_subject_*` columns are not fields.
- The root `node` lookup of `BiometricVerificationGQLType` needs 174004 as well.

`biometricMultimodalDecisions` (§6.11) reads the stored multimodal decisions under the same right.

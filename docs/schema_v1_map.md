# Schema v1 Map

This document defines a normalized SQLite schema map for Luna9 v1.

It integrates:

- Session scopes (`universal`, `local`, `tangential`)
- World state snapshots and typed deltas
- Context-construction traceability
- Memory, cognitive debt, and initiative process records
- Safety, locking, and operational audit requirements

## Design Principles

- **Append-only audit first:** keep raw turn/event history immutable.
- **State + delta first-class:** model world snapshots and transitions explicitly.
- **Traceable cognition:** every context inclusion/exclusion can be inspected.
- **Scope-aware memory:** universal/local/tangential are explicit in storage.
- **Policy-aware operations:** capture privacy/threshold/lock policies as data.

## FK Graph (High Level)

```text
projects
  └── sessions
       ├── turns
       │    ├── events
       │    ├── context_assembly_runs
       │    │    └── context_items
       │    ├── retrieval_runs
       │    │    ├── retrieval_sources
       │    │    └── retrieval_metrics
       │    └── delta_batches
       │         └── delta_ops
       ├── state_snapshots
       │    └── state_files
       ├── summaries
       ├── memory_items
       │    ├── memory_claims
       │    │    └── claim_verifications
       │    └── promotion_log
       ├── cognitive_debt
       │    └── debt_transitions
       ├── initiative_processes
       │    └── initiative_outputs
       └── locks

global_settings
  └── hmix_settings

tangential_sources
  └── memory_items (scope=tangential, via tangential_source_id)
```

## Core Identity and Session Tables

### `projects`

- `id` (PK)
- `project_key` (UNIQUE, NOT NULL)
- `git_root` (nullable)
- `anchor_root` (nullable)
- `anchor_type` (`.luna9|.l9|none`)
- `created_at`, `updated_at`

Notes:

- `project_key` resolves as git root first, then anchor root fallback.

### `sessions`

- `id` (PK)
- `project_id` (FK `projects.id`, nullable for universal)
- `scope` (`universal|local|tangential`, NOT NULL)
- `status` (`active|ended|archived`, NOT NULL)
- `started_at`, `last_active_at`, `ended_at` (nullable)
- `parent_session_id` (nullable FK `sessions.id`)

Constraints:

- One active local session per project:
  - partial unique index on (`project_id`) where
    `scope='local' AND status='active'`.

### `global_settings`

- `id` (PK, singleton)
- `content_threshold_bytes` (for inline file content)
- `sensitive_capture_policy` (JSON)
- `default_output_format` (`text|json`)
- `created_at`, `updated_at`

### `hmix_settings`

- `id` (PK)
- `session_id` (FK `sessions.id`, universal scope)
- `tone`, `detail_level`, `autonomy_level`
- `promotion_threshold` (0.0-1.0)
- `updated_at`

## Turn and Event Ledger

### `turns`

- `id` (PK)
- `project_id` (FK `projects.id`)
- `local_session_id` (FK `sessions.id`)
- `universal_session_id` (FK `sessions.id`)
- `tangential_session_id` (FK `sessions.id`)
- `mode` (`repl|oneshot`)
- `message_text` (nullable)
- `stdin_text` (nullable)
- `output_format` (`text|json`)
- `model_response` (nullable)
- `error_text` (nullable)
- `created_at`

### `events`

- `id` (PK)
- `turn_id` (FK `turns.id`)
- `event_type` (`user_input|tool_call|tool_result|model_output|system`)
- `payload_json` (NOT NULL)
- `created_at`

Notes:

- Append-only in app logic.

### `summaries`

- `id` (PK)
- `session_id` (FK `sessions.id`)
- `summary_type` (`turn_rollup|checkpoint|handoff`)
- `content` (NOT NULL)
- `source_event_start_id` (FK `events.id`)
- `source_event_end_id` (FK `events.id`)
- `created_at`

## State and Delta Model (First Class)

### `state_snapshots`

- `id` (PK)
- `project_id` (FK `projects.id`)
- `session_id` (FK `sessions.id`, local)
- `parent_state_id` (nullable FK `state_snapshots.id`)
- `capture_policy_json` (threshold/hash/ignore rules at capture time)
- `summary` (nullable)
- `created_at`

### `state_files`

- `id` (PK)
- `snapshot_id` (FK `state_snapshots.id`)
- `path` (NOT NULL)
- `file_type` (`text|binary|symlink|unknown`)
- `size_bytes`
- `is_binary` (bool)
- `content_text` (nullable, when under threshold and allowed)
- `content_hash` (NOT NULL)
- `metadata_json` (nullable)

Constraints:

- unique (`snapshot_id`, `path`).

### `delta_batches`

- `id` (PK)
- `project_id` (FK `projects.id`)
- `session_id` (FK `sessions.id`, local)
- `turn_id` (FK `turns.id`)
- `from_state_id` (FK `state_snapshots.id`)
- `to_state_id` (FK `state_snapshots.id`)
- `created_at`

### `delta_ops`

- `id` (PK)
- `batch_id` (FK `delta_batches.id`)
- `op_type` (NOT NULL)
- `target_path` (nullable)
- `payload_json` (NOT NULL)
- `inverse_payload_json` (nullable)
- `reversible` (bool, NOT NULL)
- `created_at`

Suggested `op_type` enum values:

- File: `file_created`, `file_modified`, `file_deleted`, `file_soft_deleted`
- Dependency: `dependency_added`, `dependency_removed`
- Session/context: `session_resumed`, `context_policy_changed`
- Memory/debt: `memory_promoted`, `cognitive_debt_accrued`, `debt_resolved`
- Initiative: `initiative_spawned`, `initiative_output_integrated`

## Context Construction Traceability

### `context_assembly_runs`

- `id` (PK)
- `turn_id` (FK `turns.id`)
- `strategy` (e.g. `lossy_with_lookup_v1`)
- `window_budget_tokens` (nullable)
- `notes` (nullable)
- `created_at`

### `context_items`

- `id` (PK)
- `run_id` (FK `context_assembly_runs.id`)
- `scope` (`universal|local|tangential`)
- `source_type` (`memory|summary|event|state|retrieval`)
- `source_id` (nullable)
- `decision` (`hydrated|compressed|pruned`)
- `reason_code` (NOT NULL)
- `score` (nullable)
- `blurb` (nullable)

Notes:

- This is the explainability backbone for inclusion/exclusion audits.

## Retrieval and Parametric Provenance

### `retrieval_runs`

- `id` (PK)
- `turn_id` (FK `turns.id`)
- `query_text` (nullable)
- `query_embedding_hash` (nullable)
- `method` (`surface|vector|hybrid`)
- `uv_u`, `uv_v` (nullable)
- `curvature_k`, `curvature_h` (nullable)
- `flow_suppression_applied` (bool)
- `created_at`

### `retrieval_sources`

- `id` (PK)
- `run_id` (FK `retrieval_runs.id`)
- `source_memory_id` (FK `memory_items.id`, nullable)
- `source_event_id` (FK `events.id`, nullable)
- `mode` (`smooth|exact`)
- `weight` (nullable)
- `distance` (nullable)
- `rank` (NOT NULL)

### `retrieval_metrics`

- `id` (PK)
- `run_id` (FK `retrieval_runs.id`)
- `projection_iterations` (nullable)
- `candidate_count` (nullable)
- `latency_ms` (nullable)
- `metadata_json` (nullable)

## Memory, Claims, and Promotion

### `memory_items`

- `id` (PK)
- `session_id` (FK `sessions.id`)
- `project_id` (nullable FK `projects.id`)
- `scope` (`universal|local|tangential`)
- `kind` (`fact|preference|pattern|decision|artifact|debt`)
- `content` (NOT NULL)
- `confidence` (0.0-1.0)
- `status` (`active|superseded|outdated|rejected`)
- `metadata_json` (nullable)
- `tangential_source_id` (nullable FK `tangential_sources.id`)
- `created_at`, `updated_at`

### `tangential_sources`

- `id` (PK)
- `source_type` (`doc|paper|book|repo|note|other`)
- `title` (nullable)
- `uri_or_path` (nullable)
- `attribution_json` (nullable)
- `license` (nullable)
- `created_at`

### `memory_claims`

- `id` (PK)
- `memory_id` (FK `memory_items.id`)
- `claim_text` (NOT NULL)
- `claim_type` (`factual|inferred|preference`)
- `created_at`

### `claim_verifications`

- `id` (PK)
- `claim_id` (FK `memory_claims.id`)
- `phase` (`a_priori|a_posteriori`)
- `status` (`pending|verified|falsified|needs_user_input`)
- `evidence_json` (nullable)
- `resolved_at` (nullable)
- `created_at`

### `promotion_log`

- `id` (PK)
- `source_memory_id` (FK `memory_items.id`)
- `target_memory_id` (nullable FK `memory_items.id`)
- `decision` (`auto|confirmed|rejected`)
- `reason` (nullable)
- `threshold_at_time` (NOT NULL)
- `created_at`

## Cognitive Debt and Initiative Records

### `cognitive_debt`

- `id` (PK)
- `project_id` (FK `projects.id`)
- `session_id` (FK `sessions.id`, local)
- `turn_id` (FK `turns.id`)
- `category`
  (`assumption|ambiguity|unverified_fact|inferred_intent|interaction_debt`)
- `description` (NOT NULL)
- `confidence` (0.0-1.0)
- `interest_score` (NOT NULL, default 0)
- `status` (`open|validated|falsified|expired`)
- `falsification_condition` (nullable)
- `created_at`, `resolved_at` (nullable)

### `debt_transitions`

- `id` (PK)
- `debt_id` (FK `cognitive_debt.id`)
- `from_status`, `to_status`
- `reason` (nullable)
- `created_at`

### `initiative_processes`

- `id` (PK)
- `session_id` (FK `sessions.id`)
- `project_id` (nullable FK `projects.id`)
- `process_type` (`value_extraction|reflection|curiosity`)
- `trigger_type` (NOT NULL)
- `status` (`active|paused|completed|cancelled`)
- `started_at`, `ended_at` (nullable)

### `initiative_outputs`

- `id` (PK)
- `process_id` (FK `initiative_processes.id`)
- `output_type` (`insight|question|model_update|task`)
- `content` (NOT NULL)
- `integrated_into_memory_id` (nullable FK `memory_items.id`)
- `created_at`

## Safety, Locking, and Policy Audit

### `locks`

- `id` (PK)
- `session_id` (FK `sessions.id`)
- `lock_type` (`write`)
- `owner_pid` (NOT NULL)
- `owner_host` (NOT NULL)
- `acquired_at` (NOT NULL)
- `heartbeat_at` (NOT NULL)
- `expires_at` (NOT NULL)

Constraints:

- one active write lock per session (partial unique index on
  `session_id, lock_type` where `expires_at > now`).

### `capture_policy_audit`

- `id` (PK)
- `turn_id` (FK `turns.id`)
- `policy_name` (e.g. `sensitive_denylist_v1`)
- `matched_path` (nullable)
- `action` (`excluded|redacted|allowed`)
- `reason` (nullable)
- `created_at`

## Required Index Set

- `projects(project_key)` unique
- `sessions(scope, project_id, status, last_active_at)`
- partial unique: active local session per project
- `turns(project_id, created_at)`
- `events(turn_id, created_at)`
- `state_snapshots(project_id, created_at)`
- `state_files(snapshot_id, path)` unique
- `delta_batches(project_id, created_at)`
- `delta_ops(batch_id, op_type)`
- `context_assembly_runs(turn_id)`
- `context_items(run_id, scope, decision, score)`
- `retrieval_runs(turn_id, method, created_at)`
- `retrieval_sources(run_id, mode, rank)`
- `memory_items(scope, project_id, kind, status, updated_at)`
- `memory_claims(memory_id)`
- `claim_verifications(claim_id, phase, status)`
- `promotion_log(source_memory_id, created_at)`
- `cognitive_debt(project_id, status, category, created_at)`
- `debt_transitions(debt_id, created_at)`
- `initiative_processes(session_id, process_type, status, started_at)`
- `locks(session_id, lock_type, expires_at)`
- `capture_policy_audit(turn_id, action, created_at)`

## Spec Coverage Checklist

- State snapshots + typed reversible deltas: covered
- Three context scopes (Global/Tangential/Local): covered
- Session continuity and resume: covered
- Raw event + summary persistence: covered
- Context inclusion/exclusion debuggability: covered
- Parametric retrieval provenance and geometry metadata: covered
- Cognitive debt lifecycle and interaction debt: covered
- Initiative process traces: covered
- Misconception detection and verification phases: covered
- Local-to-universal promotion with HMIX trust threshold: covered
- Single-writer locking with lease/heartbeat: covered
- Privacy boundaries and sensitive capture audit: covered

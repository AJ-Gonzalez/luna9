PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    project_key TEXT NOT NULL UNIQUE,
    git_root TEXT,
    anchor_root TEXT,
    anchor_type TEXT NOT NULL DEFAULT 'none' CHECK (
        anchor_type IN ('.luna9', '.l9', 'none')
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    scope TEXT NOT NULL CHECK (scope IN ('universal', 'local', 'tangential')),
    status TEXT NOT NULL CHECK (status IN ('active', 'ended', 'archived')),
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_active_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT,
    parent_session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS global_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    content_threshold_bytes INTEGER NOT NULL DEFAULT 8192,
    sensitive_capture_policy TEXT NOT NULL DEFAULT '{}',
    default_output_format TEXT NOT NULL DEFAULT 'text' CHECK (
        default_output_format IN ('text', 'json')
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS hmix_settings (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
    tone TEXT,
    detail_level TEXT,
    autonomy_level TEXT,
    promotion_threshold REAL NOT NULL DEFAULT 0.5 CHECK (
        promotion_threshold >= 0.0 AND promotion_threshold <= 1.0
    ),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    local_session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    universal_session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    tangential_session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    mode TEXT NOT NULL CHECK (mode IN ('repl', 'oneshot')),
    message_text TEXT,
    stdin_text TEXT,
    output_format TEXT NOT NULL DEFAULT 'text' CHECK (output_format IN ('text', 'json')),
    model_response TEXT,
    error_text TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    turn_id INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL CHECK (
        event_type IN ('user_input', 'tool_call', 'tool_result', 'model_output', 'system')
    ),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    summary_type TEXT NOT NULL CHECK (
        summary_type IN ('turn_rollup', 'checkpoint', 'handoff')
    ),
    content TEXT NOT NULL,
    source_event_start_id INTEGER REFERENCES events(id) ON DELETE SET NULL,
    source_event_end_id INTEGER REFERENCES events(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS state_snapshots (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    parent_state_id INTEGER REFERENCES state_snapshots(id) ON DELETE SET NULL,
    capture_policy_json TEXT NOT NULL DEFAULT '{}',
    summary TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS state_files (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL REFERENCES state_snapshots(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    file_type TEXT NOT NULL DEFAULT 'unknown' CHECK (
        file_type IN ('text', 'binary', 'symlink', 'unknown')
    ),
    size_bytes INTEGER,
    is_binary INTEGER NOT NULL DEFAULT 0 CHECK (is_binary IN (0, 1)),
    content_text TEXT,
    content_hash TEXT NOT NULL,
    metadata_json TEXT,
    UNIQUE (snapshot_id, path)
);

CREATE TABLE IF NOT EXISTS delta_batches (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    from_state_id INTEGER NOT NULL REFERENCES state_snapshots(id) ON DELETE RESTRICT,
    to_state_id INTEGER NOT NULL REFERENCES state_snapshots(id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS delta_ops (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL REFERENCES delta_batches(id) ON DELETE CASCADE,
    op_type TEXT NOT NULL,
    target_path TEXT,
    payload_json TEXT NOT NULL,
    inverse_payload_json TEXT,
    reversible INTEGER NOT NULL DEFAULT 0 CHECK (reversible IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS context_assembly_runs (
    id INTEGER PRIMARY KEY,
    turn_id INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    strategy TEXT NOT NULL,
    window_budget_tokens INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS context_items (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES context_assembly_runs(id) ON DELETE CASCADE,
    scope TEXT NOT NULL CHECK (scope IN ('universal', 'local', 'tangential')),
    source_type TEXT NOT NULL CHECK (
        source_type IN ('memory', 'summary', 'event', 'state', 'retrieval')
    ),
    source_id INTEGER,
    decision TEXT NOT NULL CHECK (decision IN ('hydrated', 'compressed', 'pruned')),
    reason_code TEXT NOT NULL,
    score REAL,
    blurb TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    id INTEGER PRIMARY KEY,
    turn_id INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    query_text TEXT,
    query_embedding_hash TEXT,
    method TEXT NOT NULL CHECK (method IN ('surface', 'vector', 'hybrid')),
    uv_u REAL,
    uv_v REAL,
    curvature_k REAL,
    curvature_h REAL,
    flow_suppression_applied INTEGER NOT NULL DEFAULT 0 CHECK (
        flow_suppression_applied IN (0, 1)
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retrieval_sources (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES retrieval_runs(id) ON DELETE CASCADE,
    source_memory_id INTEGER REFERENCES memory_items(id) ON DELETE SET NULL,
    source_event_id INTEGER REFERENCES events(id) ON DELETE SET NULL,
    mode TEXT NOT NULL CHECK (mode IN ('smooth', 'exact')),
    weight REAL,
    distance REAL,
    rank INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_metrics (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES retrieval_runs(id) ON DELETE CASCADE,
    projection_iterations INTEGER,
    candidate_count INTEGER,
    latency_ms REAL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS tangential_sources (
    id INTEGER PRIMARY KEY,
    source_type TEXT NOT NULL CHECK (
        source_type IN ('doc', 'paper', 'book', 'repo', 'note', 'other')
    ),
    title TEXT,
    uri_or_path TEXT,
    attribution_json TEXT,
    license TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_items (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    scope TEXT NOT NULL CHECK (scope IN ('universal', 'local', 'tangential')),
    kind TEXT NOT NULL CHECK (
        kind IN ('fact', 'preference', 'pattern', 'decision', 'artifact', 'debt')
    ),
    content TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    status TEXT NOT NULL CHECK (
        status IN ('active', 'superseded', 'outdated', 'rejected')
    ),
    metadata_json TEXT,
    tangential_source_id INTEGER REFERENCES tangential_sources(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_claims (
    id INTEGER PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    claim_type TEXT NOT NULL CHECK (claim_type IN ('factual', 'inferred', 'preference')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS claim_verifications (
    id INTEGER PRIMARY KEY,
    claim_id INTEGER NOT NULL REFERENCES memory_claims(id) ON DELETE CASCADE,
    phase TEXT NOT NULL CHECK (phase IN ('a_priori', 'a_posteriori')),
    status TEXT NOT NULL CHECK (
        status IN ('pending', 'verified', 'falsified', 'needs_user_input')
    ),
    evidence_json TEXT,
    resolved_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS promotion_log (
    id INTEGER PRIMARY KEY,
    source_memory_id INTEGER NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    target_memory_id INTEGER REFERENCES memory_items(id) ON DELETE SET NULL,
    decision TEXT NOT NULL CHECK (decision IN ('auto', 'confirmed', 'rejected')),
    reason TEXT,
    threshold_at_time REAL NOT NULL CHECK (
        threshold_at_time >= 0.0 AND threshold_at_time <= 1.0
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cognitive_debt (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id INTEGER REFERENCES turns(id) ON DELETE SET NULL,
    category TEXT NOT NULL CHECK (
        category IN (
            'assumption',
            'ambiguity',
            'unverified_fact',
            'inferred_intent',
            'interaction_debt'
        )
    ),
    description TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    interest_score REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL CHECK (status IN ('open', 'validated', 'falsified', 'expired')),
    falsification_condition TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS debt_transitions (
    id INTEGER PRIMARY KEY,
    debt_id INTEGER NOT NULL REFERENCES cognitive_debt(id) ON DELETE CASCADE,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS initiative_processes (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    process_type TEXT NOT NULL CHECK (
        process_type IN ('value_extraction', 'reflection', 'curiosity')
    ),
    trigger_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'paused', 'completed', 'cancelled')),
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS initiative_outputs (
    id INTEGER PRIMARY KEY,
    process_id INTEGER NOT NULL REFERENCES initiative_processes(id) ON DELETE CASCADE,
    output_type TEXT NOT NULL CHECK (
        output_type IN ('insight', 'question', 'model_update', 'task')
    ),
    content TEXT NOT NULL,
    integrated_into_memory_id INTEGER REFERENCES memory_items(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS locks (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    lock_type TEXT NOT NULL CHECK (lock_type IN ('write')),
    owner_pid INTEGER NOT NULL,
    owner_host TEXT NOT NULL,
    acquired_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    heartbeat_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS capture_policy_audit (
    id INTEGER PRIMARY KEY,
    turn_id INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    policy_name TEXT NOT NULL,
    matched_path TEXT,
    action TEXT NOT NULL CHECK (action IN ('excluded', 'redacted', 'allowed')),
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_scope_project_status
    ON sessions(scope, project_id, status, last_active_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_local_session
    ON sessions(project_id)
    WHERE scope = 'local' AND status = 'active';

CREATE INDEX IF NOT EXISTS idx_turns_project_created
    ON turns(project_id, created_at);

CREATE INDEX IF NOT EXISTS idx_events_turn_created
    ON events(turn_id, created_at);

CREATE INDEX IF NOT EXISTS idx_state_snapshots_project_created
    ON state_snapshots(project_id, created_at);

CREATE INDEX IF NOT EXISTS idx_delta_batches_project_created
    ON delta_batches(project_id, created_at);

CREATE INDEX IF NOT EXISTS idx_delta_ops_batch_type
    ON delta_ops(batch_id, op_type);

CREATE INDEX IF NOT EXISTS idx_context_runs_turn
    ON context_assembly_runs(turn_id);

CREATE INDEX IF NOT EXISTS idx_context_items_run_scope_decision_score
    ON context_items(run_id, scope, decision, score);

CREATE INDEX IF NOT EXISTS idx_retrieval_runs_turn_method_created
    ON retrieval_runs(turn_id, method, created_at);

CREATE INDEX IF NOT EXISTS idx_retrieval_sources_run_mode_rank
    ON retrieval_sources(run_id, mode, rank);

CREATE INDEX IF NOT EXISTS idx_memory_items_scope_project_kind_status_updated
    ON memory_items(scope, project_id, kind, status, updated_at);

CREATE INDEX IF NOT EXISTS idx_memory_claims_memory
    ON memory_claims(memory_id);

CREATE INDEX IF NOT EXISTS idx_claim_verifications_claim_phase_status
    ON claim_verifications(claim_id, phase, status);

CREATE INDEX IF NOT EXISTS idx_promotion_log_source_created
    ON promotion_log(source_memory_id, created_at);

CREATE INDEX IF NOT EXISTS idx_cognitive_debt_project_status_category_created
    ON cognitive_debt(project_id, status, category, created_at);

CREATE INDEX IF NOT EXISTS idx_debt_transitions_debt_created
    ON debt_transitions(debt_id, created_at);

CREATE INDEX IF NOT EXISTS idx_initiative_processes_session_type_status_started
    ON initiative_processes(session_id, process_type, status, started_at);

CREATE INDEX IF NOT EXISTS idx_locks_session_type_expires
    ON locks(session_id, lock_type, expires_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_active_write_lock
    ON locks(session_id, lock_type)
    WHERE is_active = 1;

CREATE INDEX IF NOT EXISTS idx_capture_policy_audit_turn_action_created
    ON capture_policy_audit(turn_id, action, created_at);

COMMIT;

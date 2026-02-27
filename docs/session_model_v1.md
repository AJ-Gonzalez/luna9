# Session Model v1

This document defines Luna9 session behavior for v1.

## Goals

- Keep a persistent universal identity across all projects.
- Keep project-specific continuity tied to the active project root.
- Include Tangential context from day one as a first-class session scope.
- Persist raw turn events for auditability and replay.

## Session Scopes

Luna9 uses three concurrent scopes every turn:

1. `universal`
   - Cross-project HMIX preferences and durable collaboration patterns.
   - Always attached.

2. `local`
   - Project-specific state, deltas, active goals, and cognitive debt.
   - One active local session per project key.

3. `tangential`
   - Domain/reference knowledge related to current work.
   - First-class in v1 (stored, ranked, and available during context assembly).

## Project Keying

Local sessions are keyed in this order:

1. Git repository root (preferred)
2. Project anchor file root (fallback): `.luna9` or `.l9`

If no git root exists, Luna9 searches upward for `.luna9` or `.l9`.

When Luna9 creates an anchor, it must add the anchor filename to `.gitignore`
if not already present.

## Session Cardinality

- One active local session per project key in v1.
- Multiple named local sessions are deferred.

## Attachment Rules

- REPL mode: attaches to `universal + local + tangential`.
- One-shot message mode (`-m/--message`): also attaches to
  `universal + local + tangential`.
- Piped stdin does not create a separate session mode; it is additional turn
  input in the same attached session scopes.

## Persistence Model

- Persist raw event log for every turn.
  - User input, tool calls, tool outputs, model output, errors, metadata.
- Persist periodic summaries/checkpoints.
- Retention policy for v1: keep all records indefinitely (manual prune only).

## Resume Behavior

- Default startup behavior auto-attaches to the last local session for the
  resolved project key.
- CLI prints a short resume banner indicating attached session identifiers.

## Locking and Concurrency

- v1 uses single-writer locking for each local session.
- If another process holds a write lock, second process must fail fast or
  degrade to read-only mode.

## Promotion Policy (Local -> Universal)

Default behavior:

- Auto-promote only clear low-risk patterns (e.g., stable format/interaction
  preferences).
- Require explicit user confirmation for deeper behavioral/value inferences.

HMIX controls this via a trust threshold:

- Users can tune promotion strictness.
- Lower strictness allows broader auto-promotion.
- Higher strictness requires confirmation more often.

## Cognitive Debt Scope

- v1: cognitive debt records are local-first.
- v2 target: add global debt aggregation/promotion.

## Privacy and Capture Boundaries

By default, capture must respect:

- `.gitignore`
- Built-in sensitive denylist, including at minimum:
  - `.env`
  - `*.pem`
  - `*.key`
  - `secrets.*`

Sensitive content must not be persisted in memory/session stores.

## Tangential Scope Requirements (v1)

Tangential is required in v1 and must support:

- Storing domain/reference artifacts separately from local project turns.
- Retrieval hooks during context assembly.
- Relevance ranking participation alongside universal and local candidates.
- Debug visibility showing when tangential items are included/excluded.

## Minimum v1 Implementation Contract

1. Resolve session key from git root or anchor fallback.
2. Attach all three scopes (`universal`, `local`, `tangential`) per turn.
3. Persist raw events for each turn plus periodic summaries.
4. Enforce single-writer lock for local session writes.
5. Auto-resume local session on startup.
6. Apply promotion policy with HMIX threshold controls.
7. Enforce privacy boundaries for capture and storage.

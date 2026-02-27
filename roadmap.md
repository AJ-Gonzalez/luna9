# Luna9 Roadmap

This roadmap is rebuilt from the tentative spec and tracks progress for the
current pivot. Items are never deleted; they are marked done when completed.

## Status Legend

- `[x]` done
- `[ ]` pending

## Immediate Pending

- [x] Define session model v1 (universal + directory-sensitive local session)
- [x] Design SQLite schema v1 for state, delta, memory, and cognitive debt
- [x] Add file operation tool layer (read/write/edit/soft-delete boundaries)
- [ ] Implement REPL session persistence and resume behavior
- [ ] Create context-construction pipeline skeleton (Global/Tangential/Local)

## Core CLI and Developer Experience

- [x] Scaffold Python project with UV packaging
- [x] Add Typer CLI entrypoint and basic command structure
- [x] Make interactive REPL the default mode
- [x] Add one-shot message mode via `-m/--message`
- [x] Support piped stdin together with one-shot message mode
- [x] Add one-shot output contract (`--output text|json`)
- [x] Set pytest + pylint + flake8 baseline and passing checks
- [ ] Add structured logging and session/event trace output
- [ ] Add robust CLI error taxonomy and user-facing recovery messages

## Session, State, and Delta Architecture

- [x] Define canonical session identity and lifecycle rules
- [ ] Implement state snapshot model for project files and metadata
- [ ] Implement typed state-delta records
- [ ] Add reversible operations where feasible (undo primitives)
- [ ] Add project/session resume and handoff hydration flow

## Context Construction

- [ ] Implement Global context store (HMIX preferences and stable identity)
- [ ] Implement Tangential context store (domain knowledge and references)
- [ ] Implement Local context store (active project/task state)
- [ ] Implement lossy-with-lookup summarization and hydration strategy
- [ ] Add relevancy ranking interface (pluggable strategies)
- [ ] Add context debug view explaining inclusion/exclusion decisions

## Memory System

- [ ] Implement SQLite-backed memory storage and retrieval APIs
- [ ] Add memory tagging, scope boundaries, and promotion rules
- [ ] Add misconception detection workflow (a priori / a posteriori)
- [ ] Add conflict resolution workflow with user confirmation
- [ ] Add memory audit/debug utilities

## Parametric Surface Retrieval

- [ ] Port/select reusable math components from legacy work into new architecture
- [ ] Implement embedding pipeline abstraction (model-configurable dimensions)
- [ ] Implement surface construction and projection in new core
- [ ] Implement dual-mode retrieval (smooth interpretation + exact provenance)
- [ ] Implement spatial/hash indexing and benchmark query scaling
- [ ] Implement flow suppression and surfaced dispersed-signal retrieval
- [ ] Add curvature-driven hydration heuristics and calibration controls

## Cognitive Debt and Initiative Engine

- [ ] Define cognitive debt object model and lifecycle states
- [ ] Ensure open debt remains hydrated in active model context
- [ ] Implement debt accrual/interest/resolution mechanics
- [ ] Implement initiative engine skeleton (value extraction, reflection,
      curiosity)
- [ ] Add trigger system for spawning initiative processes
- [ ] Expose debt and initiative state in debug output

## Subagents and Persona Architecture

- [ ] Define subagent contract (inputs, outputs, safety, lifecycle)
- [ ] Implement ephemeral persona spawning framework
- [ ] Add initial archetypes (Architect, Implementer, Reviewer, Historian,
      Curator)
- [ ] Implement merge/integration strategy for subagent outputs
- [ ] Add controls for when/why subagents are used

## Safety and Operations

- [ ] Enforce project-root and allowlist filesystem boundaries
- [ ] Keep network disabled by default with explicit enable controls
- [ ] Require confirmation for destructive or irreversible operations
- [ ] Implement secret redaction and non-persistence guardrails
- [ ] Add sandboxed subprocess execution policy where possible

## Git and Workflow Integration

- [ ] Implement git status/diff integration in CLI workflows
- [ ] Implement safe commit flow with explicit user intent gates
- [ ] Implement optional PR support workflows
- [ ] Add operation history and rollback-aware UX hooks

## Validation and Dogfooding

- [ ] Expand unit and integration test coverage for core flows
- [ ] Add benchmark harness for retrieval and context quality
- [ ] Dogfood on a real project and track breakpoints/failures
- [ ] Convert learnings into roadmap updates and refined acceptance criteria

## Documentation

- [ ] Refresh `README.md` for v1 release scope and usage

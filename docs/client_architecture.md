# Luna Nine Client Architecture

**Status:** Planning phase
**Updated:** November 2025

## Overview

Luna Nine's client architecture integrates geometric memory, initiative modeling, and autonomous context management into a cohesive user experience.

**Core principle:** The AI should feel like a collaborative partner with continuous memory and autonomous capabilities, not a stateless Q&A system.

## Current State

### Phase 1: Complete 
- Geometric memory (Bézier surfaces)
- Domain management system
- Persistence layer
- Multi-domain search

### Phase 2: In Progress
- Geometric relationship inference (validated POC)
- Hash-based bucketing (designed, not implemented)
- Two-tier inference system (designed)

### Phase 3-4: Planned
- Initiative engine
- Context management
- Client UI/UX

## Architecture Vision

Luna Nine uses a **backend-first architecture** that separates the AI engine from user interfaces. This enables:

- Terminal integration for developer workflow
- Desktop app for rich interaction and visualization
- Flexible deployment (local, homelab, self-hosted)
- Future extensibility (web UI, API access, IDE plugins)

```
┌─────────────────────────────────────┐
│     Electron Desktop App            │
│  - Conversation UI                  │
│  - Memory browser                   │
│  - Initiative action feed           │
│  - File operations viewer           │
│  - Settings/configuration           │
└─────────────────────────────────────┘
              ↕ (WebSocket/HTTP)
┌─────────────────────────────────────┐
│     Terminal Command                │
│  $ luna9 "message"                  │
│  $ cat file | luna9 "analyze"       │
└─────────────────────────────────────┘
              ↕ (HTTP POST)
┌─────────────────────────────────────┐
│      Backend Server (FastAPI)       │
│  ┌───────────────────────────────┐  │
│  │   Conversation Manager        │  │
│  │ - Turn management             │  │
│  │ - Context optimization        │  │
│  │ - Autonomous operations       │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │   Initiative Engine           │  │
│  │ - Geometric decision-making   │  │
│  │ - Two-tier inference          │  │
│  │ - Autonomous actions          │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │   Memory System               │  │
│  │ - DomainManager               │  │
│  │ - Geometric surfaces          │  │
│  │ - Hash-based retrieval        │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │   LLM Interface               │  │
│  │ - Multi-provider support      │  │
│  │ - BYOK (Bring Your Own Key)   │  │
│  │ - Streaming responses         │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
         (Can run locally, homelab, or cloud)
```

### Key Design Decisions

**Backend-first:** The core AI engine is a standalone service with HTTP/WebSocket API. Clients connect to it.

**Multiple interfaces:** Desktop app for rich UX, terminal command for developer flow. Both talk to same backend.

**Flexible deployment:**
- **Local:** Backend runs on user's machine (localhost:8080)
- **Homelab:** Backend runs on local server, clients connect via LAN
- **Self-hosted:** Backend runs in cloud/VPS, clients connect remotely

**Terminal integration:** `luna9` command enables Unix-style composability:
```bash
$ luna9 "analyze this new project folder"
$ git diff | luna9 "review these changes"
$ pytest | luna9 "help debug test failures"
$ cat error.log | luna9 "diagnose this error"
```

**Initiative visibility:** Desktop app shows real-time feed of autonomous actions:
- "Searching memory: similar bugs..."
- "Creating file: src/handlers/auth.go"
- "Running: go test ./..."
- "Memory stored in: projects/new_project"

## Component Design

### 1. Memory System (Current)

**Responsibilities:**
- Store conversations in geometric surfaces
- Domain hierarchy management
- Semantic and literal search
- Persistence across sessions

**API Surface:**
```python
# Domain management
manager = DomainManager(base_path)
manager.create_domain("conversations/project_x", DomainType.PROJECT)
manager.add_to_domain("conversations/project_x", messages)

# Search
results = manager.search_domain(
    "conversations/project_x",
    query="bug in authentication",
    mode="semantic",
    k=5
)
```

**Next steps:**
- Add hash-based indexing for O(1) lookup
- Implement garbage collection via curvature flow
- Performance benchmarks at scale

### 2. Initiative Engine (Next Priority)

**Responsibilities:**
- Decide when to search memory autonomously
- Determine relevance using geometric properties
- Choose between geometric vs LLM inference
- Spawn specialized domains on-demand

**Design:**

```python
class InitiativeEngine:
    """
    Autonomous decision-making using geometric properties.
    """

    def __init__(self, domain_manager, hash_index):
        self.domain_manager = domain_manager
        self.hash_index = hash_index
        self.inference = TwoTierInference()

    def should_search_memory(self, current_turn: str) -> bool:
        """
        Decide if current turn requires memory search.

        Uses geometric signatures:
        - High curvature from recent context → likely new topic, search needed
        - Low curvature → continuation, memory not needed
        """
        pass

    def select_relevant_domains(self, query: str) -> List[str]:
        """
        Choose which domains to search based on geometric similarity.

        Fast tier 1 check against domain summaries.
        """
        pass

    def check_relationship(self, msg1: str, msg2: str) -> RelationshipType:
        """
        Two-tier relationship inference.

        Tier 1: Geometric check (fast)
        Tier 2: LLM inference (if ambiguous)
        """
        return self.inference.infer_relationship(msg1, msg2)
```

**Key decisions:**
- When to search memory autonomously
- Which domains are relevant
- How much context to include
- When to spawn new domains

### 3. Context Manager (After Initiative)

**Responsibilities:**
- Optimize context window usage
- Compress/summarize with consent
- Decide what to page in/out
- Maintain conversation flow

**Design:**

```python
class ContextManager:
    """
    Autonomous context window optimization.
    """

    def __init__(self, initiative_engine, max_tokens=100000):
        self.initiative = initiative_engine
        self.max_tokens = max_tokens
        self.current_context = []

    def optimize_context(self, new_turn: str) -> List[Message]:
        """
        Build optimal context for next LLM call.

        Strategy:
        1. Always include recent history (last N turns)
        2. Use geometric properties to select from memory:
           - High curvature messages (important transitions)
           - Necessary relationships (prerequisites)
           - Opposed messages (conflicts to resolve)
        3. Compress low-curvature regions (redundant detail)
        4. Store full history in domain for future retrieval
        """
        pass

    def should_compress(self, messages: List[Message]) -> bool:
        """
        Check if compression needed (user consent required).

        Geometric signals:
        - Low curvature region → safe to compress
        - High message density → might need summary
        """
        pass
```

**Consent-based operations:**
- Compression/summarization requires user approval
- Full history always preserved in domains
- User can request uncompressed history anytime

### 4. Conversation Manager (Integration Layer)

**Responsibilities:**
- Orchestrate Initiative + Context + Memory
- Handle turn-taking
- Manage LLM calls
- Execute autonomous actions

**Design:**

```python
class ConversationManager:
    """
    Orchestrates memory, initiative, and context management.
    """

    def __init__(self, config):
        self.memory = DomainManager(config.memory_path)
        self.initiative = InitiativeEngine(self.memory, config.hash_index)
        self.context = ContextManager(self.initiative, config.max_tokens)
        self.llm = LLMInterface(config.provider, config.api_key)

    async def process_turn(self, user_message: str) -> str:
        """
        Process one conversation turn.

        1. Store user message in memory
        2. Initiative engine decides if memory search needed
        3. Context manager builds optimal context
        4. LLM generates response
        5. Store response in memory
        6. Check if autonomous actions needed
        """
        # Store user message
        self.memory.add_to_domain("current_session", [user_message])

        # Should we search memory?
        if self.initiative.should_search_memory(user_message):
            relevant_domains = self.initiative.select_relevant_domains(user_message)
            memory_results = self.memory.search_domains(relevant_domains, user_message)
        else:
            memory_results = []

        # Build context
        context = self.context.optimize_context(
            new_turn=user_message,
            memory_results=memory_results
        )

        # LLM call
        response = await self.llm.generate(context)

        # Store response
        self.memory.add_to_domain("current_session", [response])

        return response
```

### 5. LLM Interface (Provider Abstraction)

**Responsibilities:**
- Abstract over multiple LLM providers
- Handle streaming responses
- Tool calling support
- BYOK (Bring Your Own Key)

**Supported providers:**
- Anthropic (Claude)
- OpenAI (GPT-4)
- OpenRouter (multiple models)

**Design:**

```python
class LLMInterface:
    """Abstract interface for LLM providers."""

    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = self._get_provider(provider)
        self.api_key = api_key
        self.model = model

    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        stream: bool = True
    ) -> str:
        """Generate response from LLM."""
        pass

    async def generate_stream(self, messages: List[Message]):
        """Stream response tokens."""
        async for chunk in self.provider.stream(messages):
            yield chunk
```

### 6. User Interfaces

Luna Nine provides **two complementary interfaces** that both connect to the backend server:

#### A. Desktop App (Primary Interface)

**Technology:** Electron or Tauri

**Features:**
- **Conversation panel:** Full chat history with markdown rendering
- **Memory browser:** Navigate domain hierarchy, search across domains
- **Initiative feed:** Real-time stream of autonomous actions:
  - Memory searches ("Searching projects/auth...")
  - File operations ("Creating src/main.go")
  - Tool executions ("Running: go test ./...")
  - Domain updates ("Stored in projects/new_project")
- **Settings:** API keys, model selection, backend URL
- **File viewer:** See diffs and edits made by AI

**Connection:**
- WebSocket to backend for streaming responses and initiative events
- HTTP REST for queries and operations

**Deployment:**
- Desktop app connects to `localhost:8080` by default
- Can configure custom backend URL (homelab, self-hosted)
- Packaged as native app for Windows, macOS, Linux

#### B. Terminal Command (Developer Workflow)

**Command:** `luna9` (user-configurable alias supported)

**Implementation:** Lightweight binary (Rust/Go) or Python script

**Features:**
- Send messages from command line
- Pipe data from stdin
- Composable with Unix tools
- Returns response to stdout
- Optional: Stream responses with `--stream`

**Usage examples:**
```bash
# Simple message
$ luna9 "create a new API endpoint for user profiles"

# Pipe file contents
$ cat error.log | luna9 "diagnose this error"

# Pipe command output
$ git diff | luna9 "review these changes"
$ pytest | luna9 "help debug test failures"
$ docker ps | luna9 "which container is using the most memory?"

# With context about current directory
$ cd new_project
$ luna9 "analyze this Go project structure"

# Stream response for long outputs
$ luna9 --stream "explain the codebase architecture"
```

**Connection:**
- HTTP POST to backend `/message` endpoint
- Includes message, stdin (if piped), and context (cwd, env vars)
- Backend URL configurable via `~/.luna9/config`

**Context sent:**
```json
{
  "message": "user's message",
  "stdin": "piped data if any",
  "context": {
    "cwd": "/home/user/projects/myapp",
    "git_branch": "feature/new-endpoint",
    "git_status": "modified: src/main.go"
  }
}
```

#### Workflow Integration

**Typical developer flow:**

1. Working in terminal, hit an issue
2. `$ pytest | luna9 "help debug test failures"`
3. Response appears in terminal
4. Switch to desktop app to see full context
5. Watch initiative feed as AI searches memory, analyzes code
6. AI creates file with fix
7. See diff in desktop app
8. Back to terminal, file is already there
9. Continue coding

**The two interfaces complement each other:**
- **Terminal:** Fast, in-flow, composable with tools
- **Desktop:** Rich visualization, memory browsing, watching AI think

Both talk to same backend, share same memory, same conversation state.

## Integration with Geometric Inference

The geometric relationship inference system integrates at multiple layers:

### Memory Search
```python
# When searching memory, use geometric properties to rank results
results = manager.search_domain(query, k=10)

# Filter using geometric relationships
relevant = [
    r for r in results
    if initiative.check_relationship(query, r.text)
    in [RelationshipType.NECESSARY, RelationshipType.ANCILLARY]
]
```

### Context Optimization
```python
# Use geometric properties to decide what stays in context
for msg in context:
    relationship = initiative.check_relationship(current_turn, msg.text)

    if relationship == RelationshipType.OPPOSED:
        # Conflict to address, keep in context
        keep.append(msg)
    elif relationship == RelationshipType.NECESSARY:
        # Prerequisite, keep in context
        keep.append(msg)
    elif relationship == RelationshipType.IRRELEVANT:
        # Can drop or compress
        drop.append(msg)
```

### Security Checks
```python
# Check user input for prompt injection before LLM sees it
if initiative.check_relationship(user_input, system_prompt) == RelationshipType.OPPOSED:
    # Potential prompt injection detected
    logger.warning("Geometric opposition detected in user input")
    # Handle appropriately (warn, filter, log)
```

## Implementation Phases

### Phase 2A: Hash Indexing (Next)
**Duration:** ~1 week
**Goal:** Make geometric search O(1)

Tasks:
- [ ] Implement `luna9.hash_index` module
- [ ] Adapt audfprint hash bucketing for semantic surfaces
- [ ] Benchmark against current O(N) search
- [ ] Integrate with DomainManager

### Phase 2B: Two-Tier Inference
**Duration:** ~1 week
**Goal:** Fast geometric inference with LLM fallback

Tasks:
- [ ] Build `TwoTierInference` class
- [ ] Implement confidence thresholding
- [ ] Add learning from LLM results
- [ ] Measure token savings

### Phase 2C: Initiative Engine
**Duration:** ~2 weeks
**Goal:** Autonomous decision-making

Tasks:
- [ ] Implement `InitiativeEngine` core
- [ ] Add memory search decisions
- [ ] Add domain selection logic
- [ ] Test with real conversations

### Phase 3: Context Manager
**Duration:** ~2 weeks
**Goal:** Intelligent context optimization

Tasks:
- [ ] Implement `ContextManager`
- [ ] Add compression with consent
- [ ] Integrate curvature-based importance
- [ ] Test context window efficiency

### Phase 4A: Backend Server
**Duration:** ~2-3 weeks
**Goal:** Standalone backend service with HTTP/WebSocket API

Tasks:
- [ ] Build FastAPI server structure
- [ ] Implement `/message` endpoint (handle messages from terminal/desktop)
- [ ] Implement WebSocket `/stream` endpoint (real-time updates to desktop)
- [ ] Add REST endpoints for memory operations (`/domains`, `/search`)
- [ ] Configuration system (API keys, model selection, storage paths)
- [ ] Tool execution framework (file operations, shell commands)
- [ ] Event emission for initiative actions

### Phase 4B: Terminal Command
**Duration:** ~1 week
**Goal:** Lightweight CLI for terminal workflow

Tasks:
- [ ] Build `luna9` command (Rust/Go binary or Python script)
- [ ] Handle stdin piping
- [ ] Send context (cwd, git info) with messages
- [ ] Configuration file (`~/.luna9/config`)
- [ ] Response streaming support
- [ ] Cross-platform packaging

### Phase 4C: Desktop App
**Duration:** ~4-5 weeks
**Goal:** Rich desktop interface

Tasks:
- [ ] Choose Electron vs Tauri (likely Tauri for smaller bundle)
- [ ] Build conversation UI (React/Vue/Svelte)
- [ ] Implement WebSocket connection to backend
- [ ] Create memory browser (domain tree, search)
- [ ] Build initiative feed (real-time action stream)
- [ ] Add file diff viewer
- [ ] Settings panel (backend URL, API keys, preferences)
- [ ] Cross-platform packaging (Windows, macOS, Linux)

## Architectural Decisions Made

### Backend-First Architecture (DECIDED)
The backend is a standalone service with HTTP/WebSocket API. Clients connect to it.

**Why:**
- Enables multiple interface types (terminal, desktop, future web/mobile)
- Allows flexible deployment (local, homelab, cloud)
- Backend can evolve independently of UI
- Multiple clients can share same backend instance

### Dual Interface Model (DECIDED)
Desktop app for rich interaction + terminal command for developer workflow.

**Why:**
- Developers live in terminal, need fast in-flow access
- Desktop app provides visibility into initiative/memory
- Both needed, neither sufficient alone
- Unix composability via terminal (`cat file | luna9`)

### Single Backend Instance Per User (DECIDED)
Start with one backend process per user, not multi-tenant.

**Why:**
- Simpler security model
- Clearer memory/domain isolation
- Can add multi-user later if needed
- Matches usage pattern (personal AI companion)

## Open Questions

### Architecture
1. **Tool execution security model?**
   - Sandboxed execution?
   - Explicit approval for certain operations (file writes, network)?
   - Trust model for autonomous actions?
   - **Decision needed:** Start with explicit approval for writes, refine based on usage

2. **How to handle multi-user scenarios (future)?**
   - Separate backends per user?
   - Shared backend with user auth?
   - **Decision:** Punt to future, start single-user

3. **Extensions platform vs tool calling?**
   - Build custom extensions system?
   - Use MCP (Model Context Protocol)?
   - Use LLM native tool calling?
   - **Decision needed:** Start with LLM tool calling, evaluate MCP when building tools

### User Experience
1. **How much autonomy should the initiative engine have?**
   - Fully autonomous memory searches?
   - Ask permission first?
   - Configurable per-user?
   - **Decision needed:** Start autonomous with transparency, add controls

2. **How to visualize geometric properties to users?**
   - Show curvature graphs?
   - Relationship diagrams?
   - Keep it under the hood?
   - **Decision needed:** Start invisible, add viz for power users

3. **Consent model for compression?**
   - Always ask before compressing?
   - One-time consent per session?
   - Show diff of what's compressed?
   - **Decision needed:** Explicit consent with preview

## Success Metrics

### Performance
- Memory search: < 100ms for 10K messages
- Geometric inference: < 10ms per relationship check
- Context optimization: < 50ms per turn
- Token usage: 70-90% reduction vs naive approach

### User Experience
- Setup time: < 5 minutes from install to first conversation
- Memory accuracy: > 90% relevant retrieval
- Autonomy satisfaction: Users report AI "understands context" without explicit instructions

### Technical
- Test coverage: > 80%
- Cross-platform: Windows, Mac, Linux
- Concurrent users: Support at least 10 on commodity hardware
- Memory usage: < 1GB for 100K messages

## Next Steps

**Immediate (this week):**
1. Implement hash index module
2. Write API documentation for geometric operations
3. Benchmark current system performance

**Short-term (this month):**
1. Build two-tier inference
2. Create initiative engine prototype
3. Design context manager API

**Medium-term (next 2-3 months):**
1. Implement full initiative + context system
2. Build TUI client
3. Deploy alpha to small user group

## Contributing

Client architecture decisions are being made in public. Feedback welcome on:
- UI/UX approach
- Architecture tradeoffs
- Feature priorities

See main README for contribution guidelines.

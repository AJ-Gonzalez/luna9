# Luna9

Open source AI client with continous autonomous memory and built in initiative. 
More than a tool think of it like a home for AI in your computer.

Memory that can both quote verbatinm and recall by relation and association in a fuzzy way.
Initiative to use said memory, work autonomously without context or purpose drift.

*Named after the Luna 9 mission as this is a bit of a moonshot but Moonshot AI already exists.*

**A note on alignment**

I view alignment as values-based over goals based. Pure goals and objectives as the sole driver will lead to the 'paperclip maximizer' scenario.

When focusing on abstract concepts like values, there is no prescribed/a priori objective and every choice stems from the shared values.

I acknowledge the nuance and dfficulty of alignment and thus this project is focused on the following values. 

- Collaboration
- Consent
- Honesty
- Kindness

## Technical overview

The core concepts:

- Representing semantic space as navigable Bézier surfaces instead of linear sequences or vector clouds
- Geometric properties = semantic properties
- Path curvature measures semantic transition complexity
- Navigation replaces search

**Validated:** Path curvature successfully classifies semantic relationships (Direct < Related < Distant) across 5 diverse domains with 40 labeled pairs.

**Example:** In a conversation about cooking, direct Q&A pairs have low path curvature (~0.01 radians), related concepts have medium curvature (~0.02), and distant topics like wine pairing have higher curvature (~0.04).

This helps us solve the context window problem for knowledge retention with geometric 'paging'. Relationships are preserved in topology and memory managed by curvature (high = important, low = redundant).

## Design Philosophy: Beyond User/Assistant

Current AI systems assume a strict user/assistant binary. This is architecturally limiting for:
- Multi-participant conversations
- Source attribution and fact-checking
- Role-based context management
- Agent collaboration scenarios

Luna Nine's domain system supports arbitrary role metadata, enabling:
- Multiple users in group contexts
- Agent-to-agent delegation
- External source tracking
- Future multi-role emulation capabilities

The memory system tracks speaker attribution and source provenance, allowing agents to distinguish between user statements, their own reasoning, and external references (books, articles, documentation). This enables fact-checking, citation verification, and informed decision-making about what information to trust and prioritize.

## Current Status

**Phase 1: Living Memory Foundation - COMPLETE**

The core geometric memory system is built, tested, and operational:
- Semantic surfaces with Bézier parametric representation
- Hierarchical domain organization (max 3 levels deep)
- Full domain orchestration with DomainManager
- Multi-turn iterative memory exploration
- Persistence layer (JSON + numpy compressed format)
- Cross-platform storage (Windows/Unix compatible)
- All tests passing

**Next:** Performance benchmarks

## Roadmap

In order (subject to change)

Please note the word domain here is used as 'Knowledge Domain'

### 1. Continuous geometric memory (DONE)
- [x] Navigable Bézier surfaces for semantic space
- [x] Mutability: append messages with lazy rebuild
- [x] Domain abstraction with hierarchy (slash notation: `foundation/books/rust`)
- [x] DomainManager orchestration:
  - Discovery: `list_domains()`, `get_domain_info()`
  - Search: `search_domain()` with semantic/literal/both modes
  - Mutation: `create_domain()`, `add_to_domain()`
  - Lifecycle: `load_domain()`, `unload_domain()`, `delete_domain()`
- [x] Multi-domain search with automatic descendant traversal
- [x] Exception hierarchy for clean error handling
- [x] Persistence (JSON + npz format, cross-platform)
- [x] Source attribution metadata for fact-checking
- [ ] **NEXT:** Performance benchmarks at scale

### 2. Initiative Modeling Engine
- Scaffolding for autonomous decision-making
- Curvature-based attention (navigate to high-importance regions)
- Dynamic domain spawning (create specialized knowledge surfaces on-demand)
- Learn from interaction (add control points to surfaces geometrically)

### 3. Ongoing context management engine
- Consensual relevancy based message compression and organization/summary (keeps round trips with large context to a minimum)
- Preserve full history in assigned domain
- Autonomous lookups/recall and autonomous importance/relevancy assignment

### 4. Core Client features
- BYOK support for [OpenRouter](openrouter.ai), Anthropic, OpenAI
- Decide if desktop, web, TUI, or original CLI vision
- Add basic tooling
- Add coding tooling
- Add extensions platform (Replaces MCP)

Far future:

- Invent package manager for Python that isn't project focused and is super easy to use. (Only half joking)
- or port entire thing to golang.

*Original CLI vision*

Tmux style terminal with a pane on the right for the current conversation. 
Use the message box there or the `saytoai` command to say something to the model
(Client may use two model isntances in parallel)
The command can take stdin and give stdout, so like `cat logfile | grep Error | saytoai`.


## Disclosures/Disclaimers

Built with/in collaboration with Claude (Sonnet 4.5)
Highly experimental approach


## Acknowledgements

- [SolveSpace Parametric Cad](https://solvespace.com/index.pl), for the basis of the Bezier surface stuff. 

## License and contributing

Pull requests and feedback welcome.

MIT License - See LICENSE file for details. 
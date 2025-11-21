# Luna9

**A foundation for ethical AI interaction**

Luna Nine is three things at once:

1. **Powerful infrastructure** for any AI application - geometric memory and initiative as a service
2. **Thoughtful AI client** for developers - terminal integration meets rich desktop experience
3. **Ethical framework** for AI collaboration - stop prompting, start talking

Give LLMs continuous memory and autonomous initiative. Give users genuine collaboration instead of prompt engineering. Give both the foundation to shine.

## What Makes Luna Nine Different

**Continuous Memory:**
- Conversations persist across sessions with full context
- Geometric surfaces encode semantic relationships, not just vectors
- Memory you can navigate, not just search

**Built-in Initiative:**
- AI decides when to search memory autonomously
- Geometric inference for fast decisions (no token costs)
- Two-tier system: geometry for speed, LLM for complexity

**Beyond Prompting:**
- Terminal command for developer flow: `luna9 "thought?"`
- Desktop app to watch AI think and act
- Backend runs anywhere: localhost, homelab, cloud

*Named after the Luna 9 mission - this is a bit of a moonshot, but Moonshot AI was already taken.*

Still early stages but ethis is the vision.

## Use Luna Nine As...

### Infrastructure for Your AI App
Add geometric memory and initiative to any LLM application:

```typescript
import { Luna9 } from '@luna9/client';
import { wrapWithLuna9 } from '@luna9/ai-sdk';

const aiWithMemory = wrapWithLuna9(yourLLM, {
  apiKey: process.env.LUNA9_API_KEY,
  autoSearch: true,  // Automatically inject relevant memory
  domain: 'your-app/users/123'
});

// That's it - your AI now has continuous memory
const response = await aiWithMemory.chat(userMessage);
```

### A Thoughtful AI Client
For developers who want terminal integration and visibility into AI thinking:

```bash
# Terminal: Fast, composable, stays in flow
$ git diff | luna9 "review these changes"
$ pytest | luna9 "help debug failures"

# Desktop: Watch AI search memory, create files, make decisions
# See initiative feed, browse memory domains, configure everything
```

### Ethical Framework
Stop prompting. Start talking. Give LLMs:
- Memory that persists across sessions
- Ability to search context autonomously
- Initiative to act without being told

Give users:
- Transparent AI reasoning (watch it think)
- Control over autonomous actions
- True collaboration, not prompt engineering

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

Luna Nine is being developed as both **infrastructure** (geometric memory + initiative as a service) and a **thoughtful client** (desktop + terminal). Open source forever, with managed cloud service planned.

### Phase 1: Living Memory (COMPLETE)
- [x] Navigable Bézier surfaces for semantic space
- [x] Hierarchical domain organization (`foundation/books/rust`)
- [x] DomainManager with full CRUD operations
- [x] Semantic and literal search modes
- [x] Persistence layer (JSON + npz, cross-platform)
- [x] Source attribution for fact-checking
- [x] Geometric relationship inference (validated POC)

### Phase 2: Initiative Engine (IN PROGRESS)
- [x] Hash-based indexing for O(1) memory lookup
- [ ] Two-tier inference (geometric + LLM fallback)
- [ ] Autonomous memory search decisions
- [ ] Curvature-based attention and importance
- [ ] Dynamic domain spawning
- [ ] Geometric security (prompt injection detection)

### Phase 3: Context Management
- [ ] Intelligent context window optimization
- [ ] Consensual compression with full history preservation
- [ ] Autonomous relevancy assessment
- [ ] Token usage analytics and optimization

### Phase 4: Backend Service
- [ ] FastAPI server with REST + WebSocket
- [ ] Tool execution framework (file ops, shell commands)
- [ ] Multi-provider LLM support (Anthropic, OpenAI, OpenRouter)
- [ ] BYOK (Bring Your Own Key)
- [ ] Configuration system
- [ ] Initiative action event stream

### Phase 5: Client Applications
- [ ] **Terminal command:** `luna9 "message"` with stdin piping
- [ ] **Desktop app:** Conversation UI, memory browser, initiative feed
- [ ] Cross-platform packaging (Windows, macOS, Linux)
- [ ] Backend deployment options (localhost, homelab, cloud)

### Phase 6: Developer Experience
- [ ] **Python SDK** (native, full-featured)
- [ ] **TypeScript/JavaScript SDK** (REST wrapper)
- [ ] **Vercel AI SDK middleware** (drop-in integration)
- [ ] **CLI tools** for domain management
- [ ] Comprehensive API documentation
- [ ] Example applications and tutorials

### Phase 7: Managed Cloud Service (FUTURE)
- [ ] Luna Nine Backend as managed service
- [ ] Pay-per-use pricing with free tier
- [ ] Automatic updates and scaling
- [ ] Multi-language SDK support
- [ ] Analytics dashboard
- [ ] **Self-hosting always available** (open source commitment)

### Phase 8: Decentralized Infrastructure (FUTURE)
- [ ] **End-to-end decentralized hosting** for Luna Nine backend
- [ ] User-owned memory sovereignty (your data, your infrastructure)
- [ ] Distributed compute for memory operations
- [ ] Cryptographic verification of memory provenance
- [ ] Interoperable with traditional cloud deployment
- [ ] **Core principle:** Memory should live where you choose, not where we choose

*Building toward a future where AI memory infrastructure can be truly distributed - users control their own nodes, choose their own storage, maintain full sovereignty over their data. Self-hosting and managed services are just the beginning.*

### Long-term Vision
- Multi-user support with proper isolation
- Federated memory (share domains across instances)
- Visual memory exploration tools
- Extensions marketplace
- Enterprise on-premise deployment
- Mobile clients (iOS, Android)


## Disclosures/Disclaimers

Built with/in collaboration with Claude (Sonnet 4.5)
Highly experimental approach


## Acknowledgements

- [SolveSpace Parametric Cad](https://solvespace.com/index.pl), for the basis of the Bezier surface stuff. 
- [Audfprint](https://github.com/dpwe/audfprint), for the hashing algo to reduce complexity by a lot

## License and contributing

Pull requests and feedback welcome.

MIT License - See LICENSE file for details. 

## Other ways to support Luna9

- Leave a star!
- [Donate via Ko-fi](https://ko-fi.com/luna9project) if you'd like to support development
- Sponsor the project through GitHub
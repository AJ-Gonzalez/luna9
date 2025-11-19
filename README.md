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



## Features

WIP - proof of concept stage (it works!)



## Roadmap

In order (subject to change)

Please note the word domain here is used as 'Knowledge Domain'

1. Continous geometric memory based on n dimensional navigable Bezier surfaces. (proof of concept done)
    - Figure out update state/manage state in domains. (Add, remove/forget)
    - Start with core static domains: 
        - Personal (Collaborative relationship, user context, communication style)
        - Foundation (environment, reference books, manuals, papers, primary sources)
    - Add dynamic domain support (Domains based on projects/ languages, etc)
        - parallel awareness and traversal
    - Add resolution adjustment (Detail level per domain surface )


2. Initiative Modeling Engine
    - Scaffolding for autonomous decision-making
    - Curvature-based attention (navigate to high-importance regions)
    - Dynamic domain spawning (create specialized knowledge surfaces on-demand)
    - Learn from interaction (add control points to surfaces geometrically)

3. Ongoing context management engine
    - Consensual relevancy based message compression and organization/sumamry (keeps round trips with large context to a minimum)
    - preserve full history in assigned domain
    - autonomous lookups/recall and autonomous importance/relevancy assignment



4. Core Client features
    - BYOK support for [OpenRouter](openrouter.ai), Anthropic, OpenAI
    - Decide if desktop, web, TUI, or original CLI vision.
    - add basic tooling
    - add coding tooling
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
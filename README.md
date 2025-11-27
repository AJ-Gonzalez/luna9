# Luna 9

**AI that remembers and grows with you.**

## For Users

No more copy-pasting context. No more "remember when I said..." No more setting up projects or managing context windows.

Luna 9 just remembers. Your conversations connect across sessions. Your work compounds over time. You talk naturally and it keeps up.

## For Companies

Build AI applications with:
- **Autonomous memory** - AI decides when to search its own knowledge, no manual retrieval
- **Navigable knowledge bases** - See not just what things are, but how they're related
- **Initiative** - AI that acts on its own understanding, not just your prompts

Or use the memory layer standalone. Plug it into your existing stack. Self-host it. Scale it.

## In Practice

[Code examples coming here]

## Technical Overview

### Whitepapers

**[Parametric Surfaces for Semantic Memory](whitepapers/01_parametric_surfaces.md)** - Our core approach using Bézier surfaces for sub-linear semantic retrieval. Includes benchmarks on Project Gutenberg corpus.

More whitepapers coming - see [whitepapers/INDEX.md](whitepapers/INDEX.md) for the roadmap.

### Requirements

**For Users:**
- Python 3.11+
- `pip install luna9` (coming soon - currently install from source)

**For Development:**
```bash
cd luna9-python
pip install -e ".[dev]"
```

**Dependencies:**
- `sentence-transformers` - Semantic embeddings
- `numpy` - Numerical operations
- `scipy` - Surface math (optional, for advanced features)

See `luna9-python/pyproject.toml` for full dependency list.

## Acknowledgments

Luna9's parametric surface approach was inspired by [SolveSpace](https://solvespace.com/), an open-source parametric CAD program. Seeing how Bézier surfaces elegantly represent complex 3D geometry led us to ask: what if we applied the same mathematics to semantic space?

The spatial hash index implementation draws from [audfprint](https://github.com/dpwe/audfprint), Dan Ellis's audio fingerprinting system. The core insight - that locality-sensitive hashing enables O(1) candidate retrieval - applies beautifully to UV coordinate space.

Special thanks to both communities for building clear, well-documented open-source tools.

## Support This Work

Luna9 is open source and will remain so. If you find it useful:

- **Star the repo** - helps others discover it
- **Share your use cases** - what are you building with it?
- **Contribute** - PRs welcome for bug fixes, docs, or new features
- **Sponsor** - if Luna9 creates value for your company, consider supporting development

Building memory infrastructure for AI takes time. Your support helps us keep going.


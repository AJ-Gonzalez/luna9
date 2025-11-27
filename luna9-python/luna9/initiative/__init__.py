"""
Luna 9 Initiative Architecture

This module implements the initiative architecture where AI initiative emerges
from conditions (State, Possibilities, Boundaries) rather than being mechanistically
triggered.

Core Components:
- LMIX (Language Model Input eXperience): Translates geometric data to natural language
- State Surfacing: Current position, ambient context, trajectory
- Possibilities Mapping: High-curvature regions, tension threads
- Boundaries Framework: Values, permissions, relational context
- Initiative Engine: Orchestrates the full flow

The hypothesis: When an LLM receives geometric context rendered as natural language,
initiative emerges naturally without explicit "when to act" logic.
"""

from luna9.initiative.lmix import LMIXLexicon, LMIXTranslator
from luna9.initiative.state import StateContext, StateSurface
from luna9.initiative.boundaries import (
    Boundaries,
    BoundariesConfig,
    ChoirConfig,
    CognitionMode,
    SynthesisMethod,
)
from luna9.initiative.possibilities import Possibilities, PossibilitiesMapper
from luna9.initiative.integration import InitiativeContext, InitiativeEngine
from luna9.initiative.dual_mode import (
    DualModeRetriever,
    DualModeContext,
    create_dual_mode_retriever,
)
from luna9.initiative.flow_suppressor import FlowSuppressor, SuppressionResult

__all__ = [
    'LMIXLexicon',
    'LMIXTranslator',
    'StateContext',
    'StateSurface',
    'Boundaries',
    'BoundariesConfig',
    'ChoirConfig',
    'CognitionMode',
    'SynthesisMethod',
    'Possibilities',
    'PossibilitiesMapper',
    'InitiativeContext',
    'InitiativeEngine',
    'DualModeRetriever',
    'DualModeContext',
    'create_dual_mode_retriever',
    'FlowSuppressor',
    'SuppressionResult',
]

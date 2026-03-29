# ============================================================
# LLMRiskExplanation/__init__.py
# ============================================================
"""
LLM-Based Risk Explanation module for NeuroDrive.

Public surface:
    LLMRiskExplainer  — main orchestrator, imported by NeuroDriveUI
    RiskSnapshot      — state dataclass built from processor outputs
"""

from .llm_risk_explainer import LLMRiskExplainer
from .risk_snapshot      import RiskSnapshot

__all__ = ["LLMRiskExplainer", "RiskSnapshot"]
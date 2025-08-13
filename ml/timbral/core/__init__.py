"""
Core recommendation logic for Timbral.

This package contains the main recommendation service, scoring algorithms,
ranking logic, and explainability features for the music recommendation system.
"""

from .recommendation_service import RecommendationService
from .scoring import ScoringEngine
from .ranking import RankingEngine
from .explainability import ExplanationEngine

__all__ = ["RecommendationService", "ScoringEngine", "RankingEngine", "ExplanationEngine"] 
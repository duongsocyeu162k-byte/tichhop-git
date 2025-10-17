"""
Analytics Module
===============

This module provides analytics capabilities for the job market data.

Components:
- trend_analyzer: Analyze market trends
- salary_analyzer: Analyze salary patterns
- skills_analyzer: Analyze skills demand
"""

from .trend_analyzer import TrendAnalyzer
from .salary_analyzer import SalaryAnalyzer
from .skills_analyzer import SkillsAnalyzer

__all__ = ["TrendAnalyzer", "SalaryAnalyzer", "SkillsAnalyzer"]

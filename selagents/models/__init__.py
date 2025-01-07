"""
selagents.models module initialization

This module provides core model implementations for social-emotional learning in artificial agents,
including theory of mind capabilities, emotional memory systems, and social strategy frameworks.
"""

from .theory_of_mind import TheoryOfMind
from .emotional_memory import EmotionalMemory
from .social_strategies import (
    SocialStrategist,
    StrategyType,
    SocialAction,
    Coalition
)

__all__ = [
    'TheoryOfMind',
    'EmotionalMemory',
    'SocialStrategist',
    'StrategyType',
    'SocialAction',
    'Coalition'
]

# Version of the models module
__version__ = '0.1.0'

# Module metadata
__author__ = 'Nicolas Torres'
__email__ = 'nicolas.torresr@usm.cl'
__description__ = 'Social-Emotional Learning Models for Artificial Agents'

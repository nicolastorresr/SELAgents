"""
SELAgents
~~~~~~~~~

A comprehensive framework for implementing social and emotional learning in artificial agents,
combining reinforcement learning with emotional processing models and theory of mind capabilities.
"""

# Import version
__version__ = '0.1.0'

# Core components
from .core.agent import Agent
from .core.emotion_engine import EmotionEngine
from .core.social_network import SocialNetwork

# Models
from .models.theory_of_mind import TheoryOfMind
from .models.emotional_memory import EmotionalMemory
from .models.social_strategies import SocialStrategies

# Utilities
from .utils.metrics import MetricsCalculator
from .utils.visualizer import Visualizer

__all__ = [
    # Version
    '__version__',
    
    # Core
    'Agent',
    'EmotionEngine',
    'SocialNetwork',
    
    # Models
    'TheoryOfMind',
    'EmotionalMemory',
    'SocialStrategies',
    
    # Utils
    'MetricsCalculator',
    'Visualizer'
]

# Package metadata
__author__ = 'Nicolas Torres'
__author_email__ = 'nicolas.torresr@usm.cl'
__license__ = 'MIT'
__copyright__ = 'Copyright 2025 Nicolas Torres'
__description__ = 'A framework for social-emotional learning in artificial agents'
__url__ = 'https://github.com/nicolastorresr/SELAgents'

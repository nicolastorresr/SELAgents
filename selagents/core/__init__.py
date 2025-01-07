"""
selagents.core: Core components for social-emotional learning agents

This module provides the fundamental components for implementing agents with
social and emotional learning capabilities, including the base Agent class,
EmotionEngine for processing emotional states, and SocialNetwork for managing
agent interactions and relationships.
"""

from .agent import (
    Agent,
    EmotionalState,
    SocialBelief
)

from .emotion_engine import (
    EmotionEngine
)

from .social_network import (
    SocialNetwork,
    Relationship,
    RelationType
)

__all__ = [
    # Agent-related classes
    'Agent',
    'EmotionalState',
    'SocialBelief',
    
    # Emotion processing
    'EmotionEngine',
    
    # Social network components
    'SocialNetwork',
    'Relationship',
    'RelationType'
]

# Version information
__version__ = '0.1.0'
__author__ = 'Nicolas Torres'
__copyright__ = 'Copyright 2025'

# Module level docstring
__doc__ = """
SELAgents Core Module
====================

The core module provides the fundamental building blocks for creating and managing
agents with social and emotional learning capabilities. The main components are:

Agent
-----
The base Agent class implements social and emotional learning capabilities,
including emotional state management, decision-making, and social interaction
processing.

EmotionEngine
------------
Handles emotional processing and response generation, implementing various
emotional models and valuation mechanisms.

SocialNetwork
------------
Manages social relationships and network dynamics between agents, including
trust relationships, influence propagation, and community formation.

Example usage:
    >>> from selagents.core import Agent, SocialNetwork
    >>> 
    >>> # Create agents
    >>> agent1 = Agent(emotional_capacity=0.8)
    >>> agent2 = Agent(emotional_capacity=0.7)
    >>> 
    >>> # Create social network
    >>> network = SocialNetwork()
    >>> network.add_agent(agent1.id)
    >>> network.add_agent(agent2.id)
    >>> network.add_relationship(agent1.id, agent2.id)
"""

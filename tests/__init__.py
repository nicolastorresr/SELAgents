"""
Test suite for the SELAgents framework.
Provides comprehensive testing for agent behavior, emotional processing,
and theory of mind capabilities.
"""

from .test_agent import TestAgent
from .test_emotion_engine import TestEmotionEngine
from .test_theory_of_mind import TestTheoryOfMind

__all__ = ['TestAgent', 'TestEmotionEngine', 'TestTheoryOfMind']

# tests/test_agent.py
import unittest
import numpy as np
import torch
from selagents.core.agent import Agent
from selagents.core.emotion_engine import EmotionEngine
from selagents.models.theory_of_mind import TheoryOfMind

class TestAgent(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.agent = Agent(emotional_capacity=0.8)
        self.other_agent = Agent(emotional_capacity=0.7)
    
    def test_agent_initialization(self):
        """Test proper agent initialization with default parameters."""
        self.assertIsInstance(self.agent.emotion_engine, EmotionEngine)
        self.assertIsInstance(self.agent.theory_of_mind, TheoryOfMind)
        self.assertEqual(self.agent.emotional_capacity, 0.8)
        self.assertTrue(hasattr(self.agent, 'id'))
    
    def test_emotional_state(self):
        """Test emotional state management."""
        initial_state = self.agent.get_emotional_state()
        self.assertIsInstance(initial_state, np.ndarray)
        self.assertEqual(len(initial_state), self.agent.emotion_engine.n_emotions)
        
        # Test emotional state update
        new_state = np.array([0.5, 0.3, 0.2])  # Example emotional state
        self.agent.update_emotional_state(new_state)
        updated_state = self.agent.get_emotional_state()
        np.testing.assert_array_almost_equal(updated_state, new_state)
    
    def test_social_interaction(self):
        """Test social interaction between agents."""
        interaction = self.agent.interact(self.other_agent)
        self.assertIsInstance(interaction, dict)
        self.assertIn('emotional_impact', interaction)
        self.assertIn('social_outcome', interaction)
    
    def test_decision_making(self):
        """Test agent decision-making process."""
        context = {'social_pressure': 0.5, 'risk_level': 0.3}
        decision = self.agent.make_decision(context)
        self.assertIsInstance(decision, dict)
        self.assertIn('action', decision)
        self.assertIn('confidence', decision)
    
    def test_learning(self):
        """Test agent learning capabilities."""
        initial_knowledge = self.agent.get_knowledge_state()
        
        # Simulate learning experience
        experience = {
            'interaction': 'cooperation',
            'outcome': 'positive',
            'emotional_impact': np.array([0.6, 0.2, 0.1])
        }
        self.agent.learn(experience)
        
        updated_knowledge = self.agent.get_knowledge_state()
        self.assertNotEqual(initial_knowledge, updated_knowledge)

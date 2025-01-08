import unittest
import numpy as np
import torch
import networkx as nx
from selagents.models.theory_of_mind import TheoryOfMind

class TestTheoryOfMind(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.tom = TheoryOfMind()
        self.agent_id = 1
        self.other_agent_id = 2
    
    def test_initialization(self):
        """Test theory of mind initialization."""
        self.assertIsInstance(self.tom.mental_state_network, nx.Graph)
        self.assertTrue(hasattr(self.tom, 'belief_threshold'))
    
    def test_mental_state_prediction(self):
        """Test prediction of other agents' mental states."""
        observed_behavior = {
            'action': 'cooperate',
            'emotional_state': np.array([0.6, 0.2, 0.2])
        }
        prediction = self.tom.predict_mental_state(self.other_agent_id, observed_behavior)
        self.assertIsInstance(prediction, dict)
        self.assertIn('belief_state', prediction)
        self.assertIn('confidence', prediction)
    
    def test_belief_update(self):
        """Test belief update mechanism."""
        initial_belief = self.tom.get_belief_state(self.other_agent_id)
        
        new_observation = {
            'action': 'defect',
            'emotional_state': np.array([0.3, 0.5, 0.2])
        }
        self.tom.update_belief(self.other_agent_id, new_observation)
        
        updated_belief = self.tom.get_belief_state(self.other_agent_id)
        self.assertNotEqual(initial_belief, updated_belief)
    
    def test_social_reasoning(self):
        """Test social reasoning capabilities."""
        social_context = {
            'agents': [1, 2, 3],
            'relationships': [(1, 2, 'friendly'), (2, 3, 'neutral')]
        }
        reasoning = self.tom.reason_about_social_situation(social_context)
        self.assertIsInstance(reasoning, dict)
        self.assertIn('predicted_actions', reasoning)
        self.assertIn('confidence_levels', reasoning)
    
    def test_intention_recognition(self):
        """Test intention recognition capabilities."""
        action_sequence = [
            {'agent': 2, 'action': 'cooperate'},
            {'agent': 2, 'action': 'cooperate'},
            {'agent': 2, 'action': 'defect'}
        ]
        intention = self.tom.recognize_intention(self.other_agent_id, action_sequence)
        self.assertIsInstance(intention, dict)
        self.assertIn('predicted_intention', intention)
        self.assertIn('confidence', intention)

if __name__ == '__main__':
    unittest.main()

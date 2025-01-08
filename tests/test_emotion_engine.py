import unittest
import numpy as np
import torch
from selagents.core.emotion_engine import EmotionEngine

class TestEmotionEngine(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.emotion_engine = EmotionEngine()
    
    def test_initialization(self):
        """Test emotion engine initialization."""
        self.assertEqual(self.emotion_engine.n_emotions, 3)  # Assuming 3 basic emotions
        self.assertIsInstance(self.emotion_engine.current_state, np.ndarray)
    
    def test_emotion_processing(self):
        """Test emotional processing capabilities."""
        stimulus = np.array([0.5, 0.3, 0.2])
        response = self.emotion_engine.process_stimulus(stimulus)
        self.assertIsInstance(response, np.ndarray)
        self.assertEqual(len(response), self.emotion_engine.n_emotions)
    
    def test_emotional_memory(self):
        """Test emotional memory storage and retrieval."""
        experience = {
            'stimulus': np.array([0.4, 0.3, 0.3]),
            'response': np.array([0.5, 0.3, 0.2]),
            'outcome': 'positive'
        }
        self.emotion_engine.store_experience(experience)
        
        retrieved = self.emotion_engine.retrieve_similar_experience(
            np.array([0.45, 0.35, 0.2])
        )
        self.assertIsInstance(retrieved, dict)
    
    def test_emotion_regulation(self):
        """Test emotion regulation mechanisms."""
        initial_state = np.array([0.8, 0.1, 0.1])
        regulated_state = self.emotion_engine.regulate_emotions(initial_state)
        self.assertTrue(np.all(regulated_state >= 0))
        self.assertTrue(np.all(regulated_state <= 1))
    
    def test_emotional_influence(self):
        """Test emotional influence on decision-making."""
        decision_context = {'risk': 0.5, 'reward': 0.7}
        influence = self.emotion_engine.compute_emotional_influence(decision_context)
        self.assertIsInstance(influence, float)
        self.assertTrue(0 <= influence <= 1)

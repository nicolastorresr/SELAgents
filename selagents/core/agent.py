"""
core/agent.py: Core Agent Implementation

This module implements the base Agent class that integrates emotional processing,
social awareness, and decision-making capabilities. The agent maintains its emotional
state, processes social interactions, and adapts behavior based on experience.

Dependencies:
    - numpy: For numerical computations
    - torch: For neural network components
    - dataclasses: For structured data classes
    - typing: For type hints
    - uuid: For unique agent identification
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4

from ..models.theory_of_mind import TheoryOfMind
from ..models.emotional_memory import EmotionalMemory
from .emotion_engine import EmotionEngine
from ..utils.metrics import calculate_emotional_distance

@dataclass
class EmotionalState:
    """Represents the current emotional state of an agent."""
    # Basic emotions (values between 0 and 1)
    happiness: float = 0.5
    sadness: float = 0.0
    trust: float = 0.5
    fear: float = 0.0
    surprise: float = 0.0
    anticipation: float = 0.5
    anger: float = 0.0
    disgust: float = 0.0
    
    # Meta-emotional states
    emotional_stability: float = 0.7
    emotional_intensity: float = 0.5
    
    def to_vector(self) -> np.ndarray:
        """Convert emotional state to vector representation."""
        return np.array([
            self.happiness, self.sadness, self.trust, self.fear,
            self.surprise, self.anticipation, self.anger, self.disgust
        ])
    
    def update(self, vector: np.ndarray) -> None:
        """Update emotional state from vector representation."""
        emotions = ['happiness', 'sadness', 'trust', 'fear',
                   'surprise', 'anticipation', 'anger', 'disgust']
        for i, emotion in enumerate(emotions):
            setattr(self, emotion, float(np.clip(vector[i], 0, 1)))

class SocialBelief:
    """Represents agent's beliefs about social relationships."""
    def __init__(self, trust_level: float = 0.5, familiarity: float = 0.0):
        self.trust_level = trust_level
        self.familiarity = familiarity
        self.interaction_history: List[Dict[str, Any]] = []
        self.emotional_memory: Dict[str, float] = {}

    def update_trust(self, interaction_outcome: float) -> None:
        """Update trust level based on interaction outcome."""
        self.trust_level = (0.8 * self.trust_level + 0.2 * interaction_outcome)
        self.familiarity = min(1.0, self.familiarity + 0.1)

class Agent:
    """
    Main Agent class implementing social and emotional learning capabilities.
    """
    def __init__(
        self,
        emotional_capacity: float = 0.8,
        learning_rate: float = 0.01,
        memory_size: int = 1000,
        device: str = "cpu"
    ):
        self.id = str(uuid4())
        self.emotional_capacity = emotional_capacity
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        # Initialize components
        self.emotional_state = EmotionalState()
        self.emotion_engine = EmotionEngine(emotional_capacity)
        self.theory_of_mind = TheoryOfMind()
        self.emotional_memory = EmotionalMemory(memory_size)
        
        # Social beliefs about other agents
        self.social_beliefs: Dict[str, SocialBelief] = {}
        
        # Initialize decision network
        self._init_decision_network()
    
    def _init_decision_network(self) -> None:
        """Initialize neural network for decision making."""
        self.decision_network = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.decision_network.parameters(),
            lr=self.learning_rate
        )

    def perceive_social_context(
        self,
        other_agents: List['Agent'],
        context_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process social context and update internal state.
        
        Args:
            other_agents: List of other agents in the current context
            context_features: Additional context information
            
        Returns:
            Dict containing processed social context
        """
        social_context = {
            'agent_states': {},
            'relationship_dynamics': {},
            'environmental_factors': context_features.get('environmental', {})
        }
        
        # Process each agent in context
        for agent in other_agents:
            if agent.id not in self.social_beliefs:
                self.social_beliefs[agent.id] = SocialBelief()
            
            # Update beliefs based on theory of mind
            predicted_state = self.theory_of_mind.predict_state(agent)
            social_context['agent_states'][agent.id] = {
                'predicted_emotional_state': predicted_state,
                'trust_level': self.social_beliefs[agent.id].trust_level,
                'familiarity': self.social_beliefs[agent.id].familiarity
            }
        
        return social_context

    def process_emotion(
        self,
        stimulus: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionalState:
        """
        Process emotional response to stimulus.
        
        Args:
            stimulus: Dictionary containing stimulus information
            context: Optional context information
            
        Returns:
            Updated emotional state
        """
        # Process stimulus through emotion engine
        emotional_response = self.emotion_engine.process_stimulus(
            stimulus,
            self.emotional_state,
            context
        )
        
        # Update emotional state with stability factor
        current_state = self.emotional_state.to_vector()
        target_state = emotional_response.to_vector()
        
        stability_factor = self.emotional_state.emotional_stability
        new_state = (stability_factor * current_state +
                    (1 - stability_factor) * target_state)
        
        # Apply emotional capacity constraints
        new_state = np.clip(new_state, 0, self.emotional_capacity)
        self.emotional_state.update(new_state)
        
        # Store emotional experience in memory
        self.emotional_memory.store_experience(stimulus, self.emotional_state)
        
        return self.emotional_state

    def decide_action(
        self,
        social_context: Dict[str, Any],
        available_actions: List[str]
    ) -> Tuple[str, float]:
        """
        Decide on action based on current state and context.
        
        Args:
            social_context: Current social context
            available_actions: List of possible actions
            
        Returns:
            Tuple of (chosen_action, confidence)
        """
        # Prepare input features
        emotional_features = self.emotional_state.to_vector()
        context_features = self._encode_social_context(social_context)
        
        # Combine features
        input_features = torch.tensor(
            np.concatenate([emotional_features, context_features]),
            dtype=torch.float32,
            device=self.device
        )
        
        # Get action probabilities
        with torch.no_grad():
            action_values = self.decision_network(input_features)
            action_probs = torch.softmax(action_values, dim=0)
        
        # Select action
        action_idx = torch.argmax(action_probs).item()
        confidence = action_probs[action_idx].item()
        
        chosen_action = available_actions[action_idx]
        return chosen_action, confidence

    def interact(
        self,
        other_agent: 'Agent',
        interaction_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interact with another agent.
        
        Args:
            other_agent: The agent to interact with
            interaction_type: Type of interaction
            context: Optional interaction context
            
        Returns:
            Interaction results
        """
        # Prepare interaction context
        if context is None:
            context = {}
        
        # Update social beliefs
        if other_agent.id not in self.social_beliefs:
            self.social_beliefs[other_agent.id] = SocialBelief()
        
        # Process emotional response to interaction
        interaction_stimulus = {
            'type': interaction_type,
            'other_agent': other_agent.id,
            'context': context
        }
        emotional_response = self.process_emotion(interaction_stimulus, context)
        
        # Calculate interaction outcome
        trust_level = self.social_beliefs[other_agent.id].trust_level
        emotional_distance = calculate_emotional_distance(
            self.emotional_state,
            other_agent.emotional_state
        )
        
        interaction_success = trust_level * (1 - emotional_distance)
        
        # Update social beliefs
        self.social_beliefs[other_agent.id].update_trust(interaction_success)
        
        # Record interaction
        interaction_result = {
            'type': interaction_type,
            'success': interaction_success,
            'emotional_response': emotional_response,
            'trust_level': trust_level,
            'context': context
        }
        
        self.social_beliefs[other_agent.id].interaction_history.append(
            interaction_result
        )
        
        return interaction_result

    def learn_from_interaction(
        self,
        interaction_result: Dict[str, Any]
    ) -> None:
        """
        Learn from interaction outcome to improve future behavior.
        
        Args:
            interaction_result: Results from previous interaction
        """
        # Extract learning signals
        success = interaction_result['success']
        emotional_response = interaction_result['emotional_response']
        
        # Update emotional memory
        self.emotional_memory.update_from_interaction(interaction_result)
        
        # Update theory of mind based on interaction
        if 'other_agent' in interaction_result:
            self.theory_of_mind.update_beliefs(
                interaction_result['other_agent'],
                interaction_result
            )
        
        # Prepare training data for decision network
        state_vector = np.concatenate([
            self.emotional_state.to_vector(),
            self._encode_social_context(interaction_result['context'])
        ])
        
        # Convert to tensor
        state_tensor = torch.tensor(
            state_vector,
            dtype=torch.float32,
            device=self.device
        )
        
        # Compute target based on success
        target = torch.tensor(
            [success] * 8,
            dtype=torch.float32,
            device=self.device
        )
        
        # Update decision network
        self.optimizer.zero_grad()
        output = self.decision_network(state_tensor)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        self.optimizer.step()

    def _encode_social_context(
        self,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """
        Encode social context into feature vector.
        
        Args:
            context: Social context dictionary
            
        Returns:
            Encoded feature vector
        """
        # Extract relevant features from context
        encoded = np.zeros(8)  # Example size, adjust based on needs
        
        if 'agent_states' in context:
            # Aggregate emotional states of other agents
            states = [
                state['predicted_emotional_state']
                for state in context['agent_states'].values()
            ]
            if states:
                encoded[:4] = np.mean([s.to_vector()[:4] for s in states], axis=0)
        
        if 'environmental_factors' in context:
            # Encode environmental factors
            factors = context['environmental_factors']
            encoded[4] = factors.get('social_pressure', 0.0)
            encoded[5] = factors.get('emotional_tension', 0.0)
            encoded[6] = factors.get('group_cohesion', 0.0)
            encoded[7] = factors.get('environmental_stress', 0.0)
        
        return encoded

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return {
            'id': self.id,
            'emotional_state': self.emotional_state,
            'emotional_capacity': self.emotional_capacity,
            'social_beliefs': {
                agent_id: {
                    'trust': belief.trust_level,
                    'familiarity': belief.familiarity
                }
                for agent_id, belief in self.social_beliefs.items()
            }
        }

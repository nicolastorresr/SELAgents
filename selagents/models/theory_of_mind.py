"""
models/theory_of_mind.py: Theory of Mind Implementation

This module implements a Theory of Mind (ToM) system that enables agents to model
and predict other agents' mental states, beliefs, and intentions. It combines
bayesian belief updates with neural networks for state prediction and includes
mechanisms for tracking belief histories and uncertainty.

The implementation supports:
- Mental state prediction
- Belief modeling and updates
- Intention prediction
- Uncertainty estimation
- Social context integration
- Learning from interaction history

Dependencies:
    - numpy: For numerical computations
    - torch: For neural network components
    - dataclasses: For structured data classes
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math

@dataclass
class MentalState:
    """Represents an agent's mental state prediction."""
    # Emotional state predictions (0 to 1)
    emotional_values: Dict[str, float] = field(default_factory=lambda: {
        'happiness': 0.5,
        'sadness': 0.0,
        'trust': 0.5,
        'fear': 0.0,
        'surprise': 0.0,
        'anticipation': 0.5,
        'anger': 0.0,
        'disgust': 0.0
    })
    
    # Belief confidence (0 to 1)
    confidence: float = 0.5
    
    # Prediction uncertainty
    uncertainty: float = 0.5
    
    def to_vector(self) -> np.ndarray:
        """Convert mental state to vector representation."""
        return np.array(list(self.emotional_values.values()))

class IntentionType(Enum):
    """Types of predicted intentions."""
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    NEUTRAL = "neutral"
    SUPPORTIVE = "supportive"
    ANTAGONISTIC = "antagonistic"

@dataclass
class BeliefHistory:
    """Tracks history of beliefs about an agent."""
    mental_states: List[MentalState] = field(default_factory=list)
    interaction_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    predicted_intentions: List[IntentionType] = field(default_factory=list)
    prediction_accuracy: List[float] = field(default_factory=list)

class MentalStatePredictor(nn.Module):
    """Neural network for predicting mental states."""
    def __init__(self, input_size: int = 16, hidden_size: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 8),  # 8 emotional values
            nn.Sigmoid()
        )
        
        # Uncertainty estimation layer
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        features = self.network[2].forward(
            self.network[1].forward(
                self.network[0].forward(x)
            )
        )
        
        predictions = self.network[4].forward(
            self.network[3].forward(features)
        )
        uncertainty = self.uncertainty(features)
        
        return predictions, uncertainty

class TheoryOfMind:
    """
    Main class implementing theory of mind capabilities.
    """
    def __init__(
        self,
        learning_rate: float = 0.01,
        memory_size: int = 1000,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        
        # Initialize predictive models
        self.mental_state_predictor = MentalStatePredictor().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.mental_state_predictor.parameters(),
            lr=learning_rate
        )
        
        # Belief tracking
        self.belief_histories: Dict[str, BeliefHistory] = {}
        
        # Bayesian priors
        self.intention_priors = {
            IntentionType.COOPERATIVE: 0.3,
            IntentionType.COMPETITIVE: 0.2,
            IntentionType.NEUTRAL: 0.3,
            IntentionType.SUPPORTIVE: 0.1,
            IntentionType.ANTAGONISTIC: 0.1
        }

    def predict_state(
        self,
        target_agent: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> MentalState:
        """
        Predict mental state of target agent.
        
        Args:
            target_agent: Agent to predict state for
            context: Optional context information
            
        Returns:
            Predicted mental state
        """
        # Prepare input features
        agent_features = self._extract_agent_features(target_agent)
        context_features = self._encode_context(context)
        
        # Combine features
        input_features = torch.tensor(
            np.concatenate([agent_features, context_features]),
            dtype=torch.float32,
            device=self.device
        )
        
        # Get prediction and uncertainty
        with torch.no_grad():
            predictions, uncertainty = self.mental_state_predictor(input_features)
        
        # Convert to mental state
        emotional_values = {
            emotion: float(value)
            for emotion, value in zip(
                ['happiness', 'sadness', 'trust', 'fear',
                 'surprise', 'anticipation', 'anger', 'disgust'],
                predictions.cpu().numpy()
            )
        }
        
        mental_state = MentalState(
            emotional_values=emotional_values,
            confidence=1.0 - float(uncertainty.item()),
            uncertainty=float(uncertainty.item())
        )
        
        # Update belief history
        self._update_belief_history(target_agent.id, mental_state)
        
        return mental_state

    def update_beliefs(
        self,
        agent_id: str,
        interaction_result: Dict[str, Any]
    ) -> None:
        """
        Update beliefs based on interaction outcome.
        
        Args:
            agent_id: ID of agent to update beliefs for
            interaction_result: Interaction outcome information
        """
        if agent_id not in self.belief_histories:
            self.belief_histories[agent_id] = BeliefHistory()
        
        history = self.belief_histories[agent_id]
        
        # Record interaction outcome
        history.interaction_outcomes.append(interaction_result)
        
        # Update intention predictions
        predicted_intention = self._predict_intention(
            history.mental_states[-1] if history.mental_states else None,
            interaction_result
        )
        history.predicted_intentions.append(predicted_intention)
        
        # Calculate prediction accuracy
        if history.mental_states:
            accuracy = self._calculate_prediction_accuracy(
                history.mental_states[-1],
                interaction_result
            )
            history.prediction_accuracy.append(accuracy)
        
        # Maintain history size
        if len(history.interaction_outcomes) > self.memory_size:
            history.interaction_outcomes.pop(0)
            history.predicted_intentions.pop(0)
            history.prediction_accuracy.pop(0)
        
        # Update Bayesian priors
        self._update_intention_priors(agent_id)

    def learn(
        self,
        target_state: Dict[str, float],
        context: Dict[str, Any]
    ) -> float:
        """
        Learn from observed mental state.
        
        Args:
            target_state: Actual mental state values
            context: Context information
            
        Returns:
            Training loss
        """
        # Prepare training data
        context_features = self._encode_context(context)
        input_features = torch.tensor(
            context_features,
            dtype=torch.float32,
            device=self.device
        )
        
        target = torch.tensor(
            list(target_state.values()),
            dtype=torch.float32,
            device=self.device
        )
        
        # Training step
        self.optimizer.zero_grad()
        predictions, uncertainty = self.mental_state_predictor(input_features)
        
        # Compute loss with uncertainty regularization
        prediction_loss = nn.MSELoss()(predictions, target)
        uncertainty_regularization = 0.1 * torch.mean(uncertainty)
        loss = prediction_loss + uncertainty_regularization
        
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())

    def _extract_agent_features(
        self,
        agent: Any
    ) -> np.ndarray:
        """Extract feature vector from agent state."""
        features = np.zeros(8)
        
        # If agent has observable emotional state
        if hasattr(agent, 'emotional_state'):
            features = agent.emotional_state.to_vector()
        
        return features

    def _encode_context(
        self,
        context: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Encode context information into feature vector."""
        encoded = np.zeros(8)  # Default size
        
        if context:
            # Encode relevant context features
            encoded[0] = context.get('social_pressure', 0.0)
            encoded[1] = context.get('emotional_tension', 0.0)
            encoded[2] = context.get('group_cohesion', 0.0)
            encoded[3] = context.get('time_pressure', 0.0)
            encoded[4] = context.get('risk_level', 0.0)
            encoded[5] = context.get('uncertainty', 0.0)
            encoded[6] = context.get('cooperation_level', 0.0)
            encoded[7] = context.get('competition_level', 0.0)
        
        return encoded

    def _update_belief_history(
        self,
        agent_id: str,
        mental_state: MentalState
    ) -> None:
        """Update belief history for an agent."""
        if agent_id not in self.belief_histories:
            self.belief_histories[agent_id] = BeliefHistory()
        
        history = self.belief_histories[agent_id]
        history.mental_states.append(mental_state)
        
        # Maintain history size
        if len(history.mental_states) > self.memory_size:
            history.mental_states.pop(0)

    def _predict_intention(
        self,
        mental_state: Optional[MentalState],
        interaction_result: Dict[str, Any]
    ) -> IntentionType:
        """Predict agent's intention based on mental state and interaction."""
        if not mental_state:
            return IntentionType.NEUTRAL
        
        # Extract relevant features
        emotional_values = mental_state.emotional_values
        trust_level = emotional_values['trust']
        anger_level = emotional_values['anger']
        
        # Consider interaction success
        success = interaction_result.get('success', 0.5)
        
        # Apply Bayesian inference
        likelihoods = {
            IntentionType.COOPERATIVE: (
                0.7 * trust_level + 0.3 * success
            ),
            IntentionType.COMPETITIVE: (
                0.6 * (1 - trust_level) + 0.4 * anger_level
            ),
            IntentionType.NEUTRAL: (
                0.5 * (1 - abs(2 * success - 1))
            ),
            IntentionType.SUPPORTIVE: (
                0.8 * trust_level * success
            ),
            IntentionType.ANTAGONISTIC: (
                0.8 * anger_level * (1 - success)
            )
        }
        
        # Combine with priors
        posteriors = {
            intention: likelihood * self.intention_priors[intention]
            for intention, likelihood in likelihoods.items()
        }
        
        # Normalize
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {
                k: v/total for k, v in posteriors.items()
            }
        
        # Select most likely intention
        return max(posteriors.items(), key=lambda x: x[1])[0]

    def _calculate_prediction_accuracy(
        self,
        predicted_state: MentalState,
        interaction_result: Dict[str, Any]
    ) -> float:
        """Calculate accuracy of mental state prediction."""
        if 'actual_state' not in interaction_result:
            return 0.5  # Default when no actual state available
        
        actual_state = interaction_result['actual_state']
        predicted_vector = predicted_state.to_vector()
        actual_vector = np.array(list(actual_state.values()))
        
        # Calculate normalized accuracy
        mse = np.mean((predicted_vector - actual_vector) ** 2)
        accuracy = math.exp(-mse)  # Convert MSE to accuracy score
        
        return accuracy

    def _update_intention_priors(self, agent_id: str) -> None:
        """Update intention priors based on observation history."""
        if agent_id not in self.belief_histories:
            return
        
        history = self.belief_histories[agent_id]
        if not history.predicted_intentions:
            return
        
        # Count recent intentions
        intention_counts = {
            intention_type: 0
            for intention_type in IntentionType
        }
        
        recent_intentions = history.predicted_intentions[-100:]
        total_observations = len(recent_intentions)
        
        for intention in recent_intentions:
            intention_counts[intention] += 1
        
        # Update priors with smoothing
        smoothing_factor = 0.1
        for intention_type in IntentionType:
            count = intention_counts[intention_type]
            self.intention_priors[intention_type] = (
                (count + smoothing_factor) /
                (total_observations + smoothing_factor * len(IntentionType))
            )

    def get_belief_metrics(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """Get metrics about beliefs for an agent."""
        if agent_id not in self.belief_histories:
            return {}
        
        history = self.belief_histories[agent_id]
        
        # Calculate various metrics
        metrics = {
            'prediction_confidence': np.mean([
                state.confidence
                for state in history.mental_states[-100:]
            ]) if history.mental_states else 0.0,
            
            'prediction_accuracy': np.mean([
                acc for acc in history.prediction_accuracy[-100:]
            ]) if history.prediction_accuracy else 0.0,
            
            'intention_distribution': {
                intention_type: self.intention_priors[intention_type]
                for intention_type in IntentionType
            },
            
            'belief_stability': self._calculate_belief_stability(history),
            
            'interaction_success_rate': np.mean([
                outcome.get('success', 0.0)
                for outcome in history.interaction_outcomes[-100:]
            ]) if history.interaction_outcomes else 0.0
        }
        
        return metrics

    def _calculate_belief_stability(
        self,
        history: BeliefHistory
    ) -> float:
        """
        Calculate stability of beliefs over time.
        
        Args:
            history: Belief history for an agent
            
        Returns:
            Float between 0 and 1 indicating belief stability
        """
        if not history.mental_states or len(history.mental_states) < 2:
            return 1.0
        
        # Calculate variance in emotional predictions over time
        recent_states = history.mental_states[-20:]  # Look at last 20 states
        emotional_vectors = np.array([
            list(state.emotional_values.values())
            for state in recent_states
        ])
        
        # Calculate average variance across emotional dimensions
        state_variance = np.mean(np.var(emotional_vectors, axis=0))
        
        # Convert variance to stability score (inverse relationship)
        stability = math.exp(-state_variance)
        
        return stability

    def save_model(
        self,
        path: str
    ) -> None:
        """
        Save model state to file.
        
        Args:
            path: Path to save model state
        """
        torch.save({
            'model_state': self.mental_state_predictor.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'intention_priors': self.intention_priors,
            'belief_histories': self.belief_histories
        }, path)

    def load_model(
        self,
        path: str
    ) -> None:
        """
        Load model state from file.
        
        Args:
            path: Path to load model state from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.mental_state_predictor.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.intention_priors = checkpoint['intention_priors']
        self.belief_histories = checkpoint.get('belief_histories', {})

    def reset(self) -> None:
        """Reset all belief histories and priors to initial state."""
        self.belief_histories.clear()
        self.intention_priors = {
            intention_type: 1.0 / len(IntentionType)
            for intention_type in IntentionType
        }
        
        # Reset neural network weights
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
                
        self.mental_state_predictor.apply(weight_reset)
        
        # Reinitialize optimizer
        self.optimizer = torch.optim.Adam(
            self.mental_state_predictor.parameters(),
            lr=self.learning_rate
        )

    def get_intention_distribution(
        self,
        agent_id: str
    ) -> Dict[IntentionType, float]:
        """
        Get distribution of predicted intentions for an agent.
        
        Args:
            agent_id: ID of agent to get intentions for
            
        Returns:
            Dictionary mapping intention types to probabilities
        """
        if agent_id not in self.belief_histories:
            return {
                intention_type: 1.0 / len(IntentionType)
                for intention_type in IntentionType
            }
        
        history = self.belief_histories[agent_id]
        recent_intentions = history.predicted_intentions[-100:]
        
        if not recent_intentions:
            return self.intention_priors.copy()
        
        # Count occurrences of each intention
        intention_counts = {
            intention_type: sum(1 for i in recent_intentions if i == intention_type)
            for intention_type in IntentionType
        }
        
        # Convert to probabilities
        total_count = len(recent_intentions)
        intention_probs = {
            intention_type: count / total_count
            for intention_type, count in intention_counts.items()
        }
        
        return intention_probs

    def get_emotional_trends(
        self,
        agent_id: str,
        window_size: int = 50
    ) -> Dict[str, List[float]]:
        """
        Get emotional value trends over time.
        
        Args:
            agent_id: ID of agent to get trends for
            window_size: Number of past states to include
            
        Returns:
            Dictionary mapping emotions to lists of values
        """
        if agent_id not in self.belief_histories:
            return {}
        
        history = self.belief_histories[agent_id]
        recent_states = history.mental_states[-window_size:]
        
        if not recent_states:
            return {}
        
        # Initialize trends dictionary
        trends = {
            emotion: []
            for emotion in recent_states[0].emotional_values.keys()
        }
        
        # Collect values for each emotion
        for state in recent_states:
            for emotion, value in state.emotional_values.items():
                trends[emotion].append(value)
        
        return trends

    def merge_belief_histories(
        self,
        other_tom: 'TheoryOfMind',
        agent_ids: Optional[List[str]] = None
    ) -> None:
        """
        Merge belief histories from another TheoryOfMind instance.
        
        Args:
            other_tom: TheoryOfMind instance to merge from
            agent_ids: Optional list of specific agent IDs to merge
        """
        ids_to_merge = agent_ids if agent_ids is not None else other_tom.belief_histories.keys()
        
        for agent_id in ids_to_merge:
            if agent_id not in other_tom.belief_histories:
                continue
                
            other_history = other_tom.belief_histories[agent_id]
            
            if agent_id not in self.belief_histories:
                self.belief_histories[agent_id] = BeliefHistory()
            
            # Merge histories while maintaining size limit
            self.belief_histories[agent_id].mental_states.extend(other_history.mental_states)
            self.belief_histories[agent_id].interaction_outcomes.extend(other_history.interaction_outcomes)
            self.belief_histories[agent_id].predicted_intentions.extend(other_history.predicted_intentions)
            self.belief_histories[agent_id].prediction_accuracy.extend(other_history.prediction_accuracy)
            
            # Trim to memory size if needed
            if len(self.belief_histories[agent_id].mental_states) > self.memory_size:
                excess = len(self.belief_histories[agent_id].mental_states) - self.memory_size
                self.belief_histories[agent_id].mental_states = self.belief_histories[agent_id].mental_states[excess:]
                self.belief_histories[agent_id].interaction_outcomes = self.belief_histories[agent_id].interaction_outcomes[excess:]
                self.belief_histories[agent_id].predicted_intentions = self.belief_histories[agent_id].predicted_intentions[excess:]
                self.belief_histories[agent_id].prediction_accuracy = self.belief_histories[agent_id].prediction_accuracy[excess:]

"""
core/emotion_engine.py: Emotion Processing System

This module implements the emotional processing engine that handles emotional state
updates, stimulus processing, and emotional dynamics. It includes valuation systems,
emotional decay, and interaction effects.

Key features:
- Emotional state processing and updates
- Stimulus evaluation and emotional response generation
- Emotional decay and regulation mechanisms
- Social emotion effects (emotional contagion, empathy)
- Personality-based emotion modulation

Dependencies:
    - numpy: For numerical computations
    - scipy: For statistical functions
    - dataclasses: For structured data
    - typing: For type hints
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

class EmotionDimension(Enum):
    """Fundamental dimensions of emotion."""
    VALENCE = "valence"  # Positive vs. negative
    AROUSAL = "arousal"  # Level of activation/energy
    DOMINANCE = "dominance"  # Sense of control/power

@dataclass
class EmotionalResponse:
    """Structured emotional response to stimulus."""
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    dominance: float  # 0 to 1
    primary_emotion: str
    intensity: float  # 0 to 1
    confidence: float  # 0 to 1

class EmotionalTrait:
    """Personality-based emotional traits."""
    def __init__(
        self,
        baseline_mood: float = 0.5,
        emotional_variability: float = 0.3,
        recovery_rate: float = 0.1
    ):
        self.baseline_mood = baseline_mood
        self.emotional_variability = emotional_variability
        self.recovery_rate = recovery_rate
        
    def modulate_response(self, response: EmotionalResponse) -> EmotionalResponse:
        """Modulate emotional response based on personality traits."""
        response.intensity *= (1.0 + self.emotional_variability)
        response.valence = (
            self.baseline_mood + 
            (response.valence - self.baseline_mood) * (1.0 - self.recovery_rate)
        )
        return response

class EmotionEngine:
    """
    Main emotion processing engine implementing emotional dynamics and responses.
    """
    def __init__(
        self,
        emotional_capacity: float = 0.8,
        decay_rate: float = 0.1,
        contagion_strength: float = 0.3
    ):
        self.emotional_capacity = emotional_capacity
        self.decay_rate = decay_rate
        self.contagion_strength = contagion_strength
        
        # Initialize emotion mappings and weights
        self._init_emotion_mappings()
        
        # Personality traits
        self.emotional_trait = EmotionalTrait()
        
        # Emotional state history
        self.state_history: List[Dict[str, float]] = []
        
        # Initialize regulation mechanisms
        self._init_regulation_mechanisms()

    def _init_emotion_mappings(self) -> None:
        """Initialize emotion category mappings and relationships."""
        # Basic emotions and their dimensional representations
        self.emotion_dimensions = {
            "joy": (0.8, 0.7, 0.7),      # (valence, arousal, dominance)
            "sadness": (-0.7, 0.2, 0.3),
            "anger": (-0.6, 0.8, 0.8),
            "fear": (-0.8, 0.7, 0.2),
            "surprise": (0.1, 0.8, 0.5),
            "disgust": (-0.7, 0.5, 0.6),
            "trust": (0.6, 0.3, 0.6),
            "anticipation": (0.3, 0.6, 0.5)
        }
        
        # Emotion transition probabilities
        self.emotion_transitions = {
            "joy": {"sadness": 0.1, "trust": 0.3, "anticipation": 0.3},
            "sadness": {"anger": 0.2, "fear": 0.2, "joy": 0.1},
            "anger": {"joy": 0.1, "sadness": 0.2, "disgust": 0.3},
            "fear": {"anger": 0.2, "sadness": 0.2, "surprise": 0.2},
            "surprise": {"joy": 0.3, "fear": 0.2, "anticipation": 0.2},
            "disgust": {"anger": 0.3, "sadness": 0.2, "fear": 0.2},
            "trust": {"joy": 0.3, "anticipation": 0.2, "surprise": 0.1},
            "anticipation": {"joy": 0.3, "surprise": 0.2, "trust": 0.2}
        }

    def _init_regulation_mechanisms(self) -> None:
        """Initialize emotion regulation mechanisms."""
        self.regulation_strategies = {
            "reappraisal": self._regulate_reappraisal,
            "suppression": self._regulate_suppression,
            "acceptance": self._regulate_acceptance
        }
        
        # Default regulation thresholds
        self.regulation_thresholds = {
            "intensity": 0.8,
            "duration": 5,
            "variability": 0.4
        }

    def process_stimulus(
        self,
        stimulus: Dict[str, Any],
        current_state: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> EmotionalResponse:
        """
        Process emotional stimulus and generate response.
        
        Args:
            stimulus: Dictionary containing stimulus information
            current_state: Current emotional state
            context: Optional context information
            
        Returns:
            Generated emotional response
        """
        # Extract stimulus features
        stimulus_intensity = stimulus.get('intensity', 0.5)
        stimulus_valence = stimulus.get('valence', 0.0)
        stimulus_type = stimulus.get('type', 'neutral')
        
        # Consider context
        context_factors = self._evaluate_context(context)
        
        # Generate base response
        base_response = self._generate_base_response(
            stimulus_type,
            stimulus_intensity,
            stimulus_valence
        )
        
        # Apply context modulation
        modulated_response = self._modulate_response(base_response, context_factors)
        
        # Apply personality traits
        final_response = self.emotional_trait.modulate_response(modulated_response)
        
        # Apply emotional capacity constraint
        final_response = self._apply_capacity_constraint(final_response)
        
        # Update state history
        self._update_history(final_response)
        
        return final_response

    def _generate_base_response(
        self,
        stimulus_type: str,
        intensity: float,
        valence: float
    ) -> EmotionalResponse:
        """Generate base emotional response to stimulus."""
        # Map stimulus to emotion dimensions
        emotion_dims = self._map_stimulus_to_dimensions(stimulus_type, valence)
        
        # Calculate primary emotion
        primary_emotion = self._determine_primary_emotion(emotion_dims)
        
        # Generate response with uncertainty
        confidence = self._calculate_response_confidence(emotion_dims, intensity)
        
        return EmotionalResponse(
            valence=emotion_dims[0],
            arousal=emotion_dims[1],
            dominance=emotion_dims[2],
            primary_emotion=primary_emotion,
            intensity=intensity,
            confidence=confidence
        )

    def _map_stimulus_to_dimensions(
        self,
        stimulus_type: str,
        valence: float
    ) -> Tuple[float, float, float]:
        """Map stimulus to emotional dimensions."""
        # Get base dimensional mapping for stimulus type
        base_dims = self.emotion_dimensions.get(
            stimulus_type,
            (valence, 0.5, 0.5)  # Default if type not found
        )
        
        # Add noise for natural variation
        noise = np.random.normal(0, 0.1, 3)
        dims = tuple(np.clip(np.array(base_dims) + noise, -1, 1))
        
        return dims

    def _determine_primary_emotion(
        self,
        emotion_dims: Tuple[float, float, float]
    ) -> str:
        """Determine primary emotion from dimensional values."""
        # Calculate distance to each emotion prototype
        distances = {
            emotion: np.linalg.norm(
                np.array(emotion_dims) - np.array(dims)
            )
            for emotion, dims in self.emotion_dimensions.items()
        }
        
        # Return closest emotion
        return min(distances.items(), key=lambda x: x[1])[0]

    def _calculate_response_confidence(
        self,
        emotion_dims: Tuple[float, float, float],
        intensity: float
    ) -> float:
        """Calculate confidence in emotional response."""
        # Consider distance to prototype emotions
        distances = [
            np.linalg.norm(np.array(emotion_dims) - np.array(dims))
            for dims in self.emotion_dimensions.values()
        ]
        
        # Confidence decreases with ambiguity (similar distances to multiple emotions)
        distinctiveness = stats.variation(distances)
        
        # Confidence increases with intensity
        confidence = 0.5 + (0.3 * intensity) + (0.2 * distinctiveness)
        return np.clip(confidence, 0, 1)

    def _evaluate_context(
        self,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate contextual factors affecting emotion."""
        if context is None:
            return {"social": 0.0, "environmental": 0.0, "temporal": 0.0}
        
        factors = {
            "social": self._evaluate_social_context(context),
            "environmental": self._evaluate_environmental_context(context),
            "temporal": self._evaluate_temporal_context(context)
        }
        
        return factors

    def _evaluate_social_context(self, context: Dict[str, Any]) -> float:
        """Evaluate social factors in context."""
        social_factors = context.get('social', {})
        
        # Consider social presence
        social_presence = social_factors.get('presence', 0.0)
        
        # Consider relationship factors
        relationship = social_factors.get('relationship', {})
        familiarity = relationship.get('familiarity', 0.0)
        trust = relationship.get('trust', 0.5)
        
        # Combine factors
        social_impact = (0.4 * social_presence +
                        0.3 * familiarity +
                        0.3 * trust)
        
        return social_impact

    def _evaluate_environmental_context(self, context: Dict[str, Any]) -> float:
        """Evaluate environmental factors in context."""
        env_factors = context.get('environmental', {})
        
        # Consider physical environment
        safety = env_factors.get('safety', 0.5)
        comfort = env_factors.get('comfort', 0.5)
        
        # Consider situational factors
        urgency = env_factors.get('urgency', 0.0)
        
        # Combine factors
        env_impact = (0.4 * safety +
                     0.3 * comfort +
                     0.3 * (1 - urgency))
        
        return env_impact

    def _evaluate_temporal_context(self, context: Dict[str, Any]) -> float:
        """Evaluate temporal factors in context."""
        temporal_factors = context.get('temporal', {})
        
        # Consider time-based factors
        duration = temporal_factors.get('duration', 0.0)
        frequency = temporal_factors.get('frequency', 0.0)
        
        # Combine factors
        temporal_impact = 0.6 * duration + 0.4 * frequency
        
        return temporal_impact

    def _modulate_response(
        self,
        response: EmotionalResponse,
        context_factors: Dict[str, float]
    ) -> EmotionalResponse:
        """Modulate emotional response based on context."""
        # Calculate context weights
        social_weight = 0.4
        environmental_weight = 0.3
        temporal_weight = 0.3
        
        # Modulate intensity
        context_modifier = (
            social_weight * context_factors['social'] +
            environmental_weight * context_factors['environmental'] +
            temporal_weight * context_factors['temporal']
        )
        
        response.intensity *= (1.0 + context_modifier)
        response.intensity = np.clip(response.intensity, 0, 1)
        
        return response

    def _apply_capacity_constraint(
        self,
        response: EmotionalResponse
    ) -> EmotionalResponse:
        """Apply emotional capacity constraints to response."""
        response.intensity = min(response.intensity, self.emotional_capacity)
        response.arousal = min(response.arousal, self.emotional_capacity)
        
        return response

    def _update_history(self, response: EmotionalResponse) -> None:
        """Update emotional state history."""
        state_entry = {
            'valence': response.valence,
            'arousal': response.arousal,
            'dominance': response.dominance,
            'intensity': response.intensity,
            'emotion': response.primary_emotion
        }
        
        self.state_history.append(state_entry)
        
        # Maintain limited history
        if len(self.state_history) > 100:
            self.state_history.pop(0)

    def _regulate_reappraisal(
        self,
        response: EmotionalResponse
    ) -> EmotionalResponse:
        """Apply cognitive reappraisal regulation strategy."""
        if response.intensity > self.regulation_thresholds['intensity']:
            response.intensity *= 0.7
            response.arousal *= 0.8
        return response

    def _regulate_suppression(
        self,
        response: EmotionalResponse
    ) -> EmotionalResponse:
        """Apply emotional suppression regulation strategy."""
        if response.arousal > self.regulation_thresholds['intensity']:
            response.arousal *= 0.6
            response.intensity *= 0.9
        return response

    def _regulate_acceptance(
        self,
        response: EmotionalResponse
    ) -> EmotionalResponse:
        """Apply emotional acceptance regulation strategy."""
        if response.intensity > self.regulation_thresholds['intensity']:
            response.dominance *= 1.2
        return response

    def apply_emotional_contagion(
        self,
        response: EmotionalResponse,
        other_responses: List[EmotionalResponse]
    ) -> EmotionalResponse:
        """Apply emotional contagion effects from other agents."""
        if not other_responses:
            return response
        
        # Calculate average emotional state of others
        other_valence = np.mean([r.valence for r in other_responses])
        other_arousal = np.mean([r.arousal for r in other_responses])
        
        # Apply contagion effect
        response.valence = (
            (1 - self.contagion_strength) * response.valence +
            self.contagion_strength * other_valence
        )
        
        response.arousal = (
            (1 - self.contagion_strength) * response.arousal +
            self.contagion_strength * other_arousal
        )
        
        return response

    def get_emotional_dynamics(
        self,
        window_size: int = 10
    ) -> Dict[str, List[float]]:
        """Get emotional dynamics over recent history.
        
        Args:
            window_size: Number of recent states to analyze
            
        Returns:
            Dictionary containing emotional dynamics metrics
        """
        if len(self.state_history) < 2:
            return {
                'valence_trend': [],
                'arousal_trend': [],
                'intensity_trend': [],
                'emotional_stability': 0.0,
                'dominant_emotions': []
            }
        
        # Get recent history window
        recent_states = self.state_history[-window_size:]
        
        # Calculate trends
        valence_trend = [state['valence'] for state in recent_states]
        arousal_trend = [state['arousal'] for state in recent_states]
        intensity_trend = [state['intensity'] for state in recent_states]
        
        # Calculate stability (inverse of standard deviation)
        stability = 1.0 - min(1.0, np.std(intensity_trend))
        
        # Get dominant emotions
        emotions = [state['emotion'] for state in recent_states]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotions = sorted(
            emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        return {
            'valence_trend': valence_trend,
            'arousal_trend': arousal_trend,
            'intensity_trend': intensity_trend,
            'emotional_stability': stability,
            'dominant_emotions': dominant_emotions
        }

    def decay_emotional_state(self, time_elapsed: float) -> None:
        """Apply emotional decay over time.
        
        Args:
            time_elapsed: Time elapsed since last update
        """
        if not self.state_history:
            return
        
        current_state = self.state_history[-1]
        decay_factor = np.exp(-self.decay_rate * time_elapsed)
        
        decayed_state = {
            'valence': current_state['valence'] * decay_factor,
            'arousal': current_state['arousal'] * decay_factor,
            'dominance': current_state['dominance'],
            'intensity': current_state['intensity'] * decay_factor,
            'emotion': current_state['emotion']
        }
        
        self._update_history(EmotionalResponse(
            valence=decayed_state['valence'],
            arousal=decayed_state['arousal'],
            dominance=decayed_state['dominance'],
            primary_emotion=decayed_state['emotion'],
            intensity=decayed_state['intensity'],
            confidence=1.0
        ))

    def regulate_emotion(
        self,
        response: EmotionalResponse,
        strategy: str = "reappraisal"
    ) -> EmotionalResponse:
        """Apply emotion regulation strategy.
        
        Args:
            response: Current emotional response
            strategy: Regulation strategy to apply
            
        Returns:
            Regulated emotional response
        """
        if strategy not in self.regulation_strategies:
            return response
        
        return self.regulation_strategies[strategy](response)

    def get_current_state(self) -> Optional[Dict[str, float]]:
        """Get current emotional state."""
        if not self.state_history:
            return None
        return self.state_history[-1]

    def reset_state(self) -> None:
        """Reset emotional state to neutral."""
        neutral_response = EmotionalResponse(
            valence=0.0,
            arousal=0.5,
            dominance=0.5,
            primary_emotion="neutral",
            intensity=0.5,
            confidence=1.0
        )
        self.state_history = []
        self._update_history(neutral_response)

    def get_transition_probability(
        self,
        current_emotion: str,
        target_emotion: str
    ) -> float:
        """Get probability of transition between emotions.
        
        Args:
            current_emotion: Starting emotion
            target_emotion: Target emotion
            
        Returns:
            Transition probability
        """
        if current_emotion not in self.emotion_transitions:
            return 0.0
        return self.emotion_transitions[current_emotion].get(target_emotion, 0.0)

    def simulate_emotional_dynamics(
        self,
        steps: int,
        base_intensity: float = 0.5
    ) -> List[EmotionalResponse]:
        """Simulate emotional dynamics over time.
        
        Args:
            steps: Number of simulation steps
            base_intensity: Base emotional intensity
            
        Returns:
            List of emotional responses over time
        """
        responses = []
        current_emotion = "neutral"
        
        for _ in range(steps):
            # Generate random stimulus
            stimulus = {
                'type': current_emotion,
                'intensity': base_intensity + np.random.normal(0, 0.1),
                'valence': self.emotion_dimensions[current_emotion][0]
            }
            
            # Process stimulus
            response = self.process_stimulus(stimulus, None)
            responses.append(response)
            
            # Determine next emotion based on transition probabilities
            if current_emotion in self.emotion_transitions:
                transitions = self.emotion_transitions[current_emotion]
                emotions = list(transitions.keys())
                probabilities = list(transitions.values())
                
                if np.random.random() < sum(probabilities):
                    current_emotion = np.random.choice(
                        emotions,
                        p=np.array(probabilities) / sum(probabilities)
                    )
        
        return responses

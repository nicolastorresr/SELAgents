"""
models/emotional_memory.py: Emotional Memory System

This module implements an emotional memory system that enables agents to learn from
and recall past emotional experiences. It combines episodic memory storage with
emotional valuation and retrieval mechanisms.

The implementation supports:
- Storage of emotional experiences
- Experience valuation based on emotional intensity
- Context-aware memory retrieval
- Emotional pattern learning
- Memory consolidation and forgetting
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time
from collections import deque
import heapq

@dataclass
class EmotionalExperience:
    """Represents a single emotional experience or memory."""
    # Unique identifier
    id: str
    # Timestamp of experience
    timestamp: float
    # Emotional state during experience
    emotional_state: Dict[str, float]
    # Context of the experience
    context: Dict[str, Any]
    # Valence (-1 to 1) representing positive/negative nature
    valence: float
    # Arousal (0 to 1) representing intensity
    arousal: float
    # Dominance (0 to 1) representing sense of control
    dominance: float
    # Associated reward or outcome
    reward: float = 0.0
    # Memory strength (decays over time)
    strength: float = 1.0
    # Tags for categorical organization
    tags: List[str] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Convert experience to vector representation."""
        emotion_vector = np.array(list(self.emotional_state.values()))
        context_vector = np.array([
            self.context.get('social_pressure', 0.0),
            self.context.get('environmental_stress', 0.0),
            self.context.get('cognitive_load', 0.0)
        ])
        return np.concatenate([
            emotion_vector,
            context_vector,
            [self.valence, self.arousal, self.dominance, self.reward]
        ])

class MemoryType(Enum):
    """Types of emotional memories."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SIGNIFICANT = "significant"

class EmotionalPatternNetwork(nn.Module):
    """Neural network for learning emotional patterns."""
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self.dominance_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning valence, arousal, and dominance predictions."""
        features = self.encoder(x)
        valence = self.valence_head(features)
        arousal = self.arousal_head(features)
        dominance = self.dominance_head(features)
        return valence, arousal, dominance

class EmotionalMemory:
    """Main class implementing emotional memory system."""
    
    def __init__(
        self,
        memory_capacity: int = 10000,
        learning_rate: float = 0.001,
        decay_rate: float = 0.1,
        consolidation_threshold: float = 0.7,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.memory_capacity = memory_capacity
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold
        
        # Memory storage
        self.short_term_memory: deque = deque(maxlen=100)
        self.long_term_memory: List[EmotionalExperience] = []
        self.significant_memory: List[EmotionalExperience] = []
        
        # Pattern learning network
        self.pattern_network = EmotionalPatternNetwork(
            input_size=15  # Adjust based on vector size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.pattern_network.parameters(),
            lr=learning_rate
        )
        
        # Index structures for efficient retrieval
        self.emotion_index: Dict[str, List[EmotionalExperience]] = {}
        self.temporal_index: List[Tuple[float, EmotionalExperience]] = []
        self.context_index: Dict[str, List[EmotionalExperience]] = {}

    def store_experience(
        self,
        emotional_state: Dict[str, float],
        context: Dict[str, Any],
        reward: float,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store new emotional experience."""
        # Create experience
        experience = EmotionalExperience(
            id=f"exp_{time.time()}",
            timestamp=time.time(),
            emotional_state=emotional_state,
            context=context,
            reward=reward,
            tags=tags or [],
            valence=self._calculate_valence(emotional_state),
            arousal=self._calculate_arousal(emotional_state),
            dominance=self._calculate_dominance(emotional_state)
        )
        
        # Store in short-term memory
        self.short_term_memory.append(experience)
        
        # Update indices
        self._update_indices(experience)
        
        # Consider consolidation
        self._consider_consolidation(experience)
        
        return experience.id

    def retrieve_similar(
        self,
        emotional_state: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[EmotionalExperience]:
        """Retrieve k most similar experiences."""
        query_vector = self._create_query_vector(emotional_state, context)
        
        # Compute similarities across all memories
        memories = (
            list(self.short_term_memory) +
            self.long_term_memory +
            self.significant_memory
        )
        
        if not memories:
            return []
        
        # Calculate similarities and apply decay
        similarities = []
        current_time = time.time()
        
        for memory in memories:
            memory_vector = memory.to_vector()
            similarity = self._compute_similarity(query_vector, memory_vector)
            
            # Apply time decay
            time_factor = np.exp(-self.decay_rate * (current_time - memory.timestamp))
            similarity *= time_factor
            
            similarities.append((similarity, memory))
        
        # Return top k
        top_k = heapq.nlargest(k, similarities, key=lambda x: x[0])
        return [memory for _, memory in top_k]

    def update_memory_strength(
        self,
        experience_id: str,
        reinforcement: float
    ) -> None:
        """Update memory strength based on reinforcement."""
        for memory_list in [self.short_term_memory, self.long_term_memory, self.significant_memory]:
            for memory in memory_list:
                if memory.id == experience_id:
                    memory.strength = min(1.0, memory.strength + reinforcement)
                    break

    def consolidate_memory(self) -> None:
        """Consolidate short-term memories into long-term storage."""
        current_time = time.time()
        
        # Check each short-term memory for consolidation
        for experience in list(self.short_term_memory):
            # Skip recent memories
            if current_time - experience.timestamp < 300:  # 5 minutes
                continue
            
            if experience.strength >= self.consolidation_threshold:
                if experience.arousal > 0.8:  # High emotional intensity
                    self.significant_memory.append(experience)
                else:
                    self.long_term_memory.append(experience)
                
                self.short_term_memory.remove(experience)
        
        # Maintain capacity limits
        while len(self.long_term_memory) > self.memory_capacity:
            self._forget_weakest_memory()

    def learn_patterns(
        self,
        batch_size: int = 32
    ) -> float:
        """Learn emotional patterns from stored experiences."""
        if len(self.long_term_memory) < batch_size:
            return 0.0
        
        # Sample batch of experiences
        batch = np.random.choice(self.long_term_memory, batch_size)
        
        # Prepare training data
        input_vectors = torch.FloatTensor(
            [exp.to_vector() for exp in batch]
        ).to(self.device)
        
        targets = torch.FloatTensor([
            [exp.valence, exp.arousal, exp.dominance]
            for exp in batch
        ]).to(self.device)
        
        # Training step
        self.optimizer.zero_grad()
        valence_pred, arousal_pred, dominance_pred = self.pattern_network(input_vectors)
        
        # Compute loss
        loss = (
            nn.MSELoss()(valence_pred, targets[:, 0:1]) +
            nn.MSELoss()(arousal_pred, targets[:, 1:2]) +
            nn.MSELoss()(dominance_pred, targets[:, 2:3])
        )
        
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())

    def _calculate_valence(self, emotional_state: Dict[str, float]) -> float:
        """Calculate emotional valence from state."""
        positive_emotions = ['joy', 'trust', 'anticipation']
        negative_emotions = ['fear', 'sadness', 'disgust', 'anger']
        
        positive_value = sum(emotional_state.get(emotion, 0.0) for emotion in positive_emotions)
        negative_value = sum(emotional_state.get(emotion, 0.0) for emotion in negative_emotions)
        
        return np.tanh(positive_value - negative_value)

    def _calculate_arousal(self, emotional_state: Dict[str, float]) -> float:
        """Calculate emotional arousal from state."""
        high_arousal = ['joy', 'fear', 'anger', 'surprise']
        return min(1.0, sum(emotional_state.get(emotion, 0.0) for emotion in high_arousal))

    def _calculate_dominance(self, emotional_state: Dict[str, float]) -> float:
        """Calculate emotional dominance from state."""
        dominant_emotions = ['joy', 'anger', 'anticipation']
        submissive_emotions = ['fear', 'sadness']
        
        dominant_value = sum(emotional_state.get(emotion, 0.0) for emotion in dominant_emotions)
        submissive_value = sum(emotional_state.get(emotion, 0.0) for emotion in submissive_emotions)
        
        return 1.0 / (1.0 + np.exp(submissive_value - dominant_value))

    def _update_indices(self, experience: EmotionalExperience) -> None:
        """Update memory indices with new experience."""
        # Update emotion index
        for emotion, intensity in experience.emotional_state.items():
            if intensity > 0.5:  # Only index significant emotions
                if emotion not in self.emotion_index:
                    self.emotion_index[emotion] = []
                self.emotion_index[emotion].append(experience)
        
        # Update temporal index
        heapq.heappush(self.temporal_index, (experience.timestamp, experience))
        
        # Update context index
        for context_key, context_value in experience.context.items():
            key = f"{context_key}:{context_value}"
            if key not in self.context_index:
                self.context_index[key] = []
            self.context_index[key].append(experience)

    def _forget_weakest_memory(self) -> None:
        """Remove the weakest memory from long-term storage."""
        if not self.long_term_memory:
            return
        
        # Find weakest memory
        weakest_idx = min(
            range(len(self.long_term_memory)),
            key=lambda i: self.long_term_memory[i].strength
        )
        
        # Remove from indices
        forgotten = self.long_term_memory.pop(weakest_idx)
        self._remove_from_indices(forgotten)

    def _remove_from_indices(self, experience: EmotionalExperience) -> None:
        """Remove experience from all indices."""
        # Remove from emotion index
        for emotions in self.emotion_index.values():
            if experience in emotions:
                emotions.remove(experience)
        
        # Remove from temporal index
        self.temporal_index = [
            (t, e) for t, e in self.temporal_index
            if e.id != experience.id
        ]
        heapq.heapify(self.temporal_index)
        
        # Remove from context index
        for contexts in self.context_index.values():
            if experience in contexts:
                contexts.remove(experience)

    def _consider_consolidation(self, experience: EmotionalExperience) -> None:
        """Consider immediate consolidation for highly significant experiences."""
        if (experience.arousal > 0.9 and abs(experience.valence) > 0.8):
            # Highly emotional experience - consolidate immediately
            self.significant_memory.append(experience)
            self.short_term_memory.remove(experience)

    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two experience vectors."""
        return 1.0 / (1.0 + np.linalg.norm(vec1 - vec2))

    def _create_query_vector(
        self,
        emotional_state: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Create vector representation for memory query."""
        context = context or {}
        emotion_vector = np.array(list(emotional_state.values()))
        context_vector = np.array([
            context.get('social_pressure', 0.0),
            context.get('environmental_stress', 0.0),
            context.get('cognitive_load', 0.0)
        ])
        
        valence = self._calculate_valence(emotional_state)
        arousal = self._calculate_arousal(emotional_state)
        dominance = self._calculate_dominance(emotional_state)
        
        return np.concatenate([
            emotion_vector,
            context_vector,
            [valence, arousal, dominance, 0.0]  # 0.0 for reward placeholder
        ])

    def save_state(self, path: str) -> None:
        """Save memory state and model to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save memories
        memories_data = {
            'short_term': [exp.to_dict() for exp in self.short_term_memory],
            'long_term': [exp.to_dict() for exp in self.long_term_memory],
            'significant': [exp.to_dict() for exp in self.significant_memory]
        }
        
        with open(save_path / 'memories.json', 'w') as f:
            json.dump(memories_data, f)
        
        # Save model state
        torch.save({
            'model_state': self.pattern_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'params': {
                'memory_capacity': self.memory_capacity,
                'learning_rate': self.learning_rate,
                'decay_rate': self.decay_rate,
                'consolidation_threshold': self.consolidation_threshold
            }
        }, save_path / 'model.pt')

    def load_state(self, path: str) -> None:
        """Load memory state and model from disk."""
        load_path = Path(path)
        
        # Load memories
        with open(load_path / 'memories.json', 'r') as f:
            memories_data = json.load(f)
        
        self.short_term_memory = deque(
            [EmotionalExperience.from_dict(exp) for exp in memories_data['short_term']],
            maxlen=100
        )
        self.long_term_memory = [
            EmotionalExperience.from_dict(exp) for exp in memories_data['long_term']
        ]
        self.significant_memory = [
            EmotionalExperience.from_dict(exp) for exp in memories_data['significant']
        ]
        
        # Load model state
        checkpoint = torch.load(load_path / 'model.pt', map_location=self.device)
        self.pattern_network.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Restore parameters
        params = checkpoint['params']
        self.memory_capacity = params['memory_capacity']
        self.learning_rate = params['learning_rate']
        self.decay_rate = params['decay_rate']
        self.consolidation_threshold = params['consolidation_threshold']
        
        # Rebuild indices
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild memory indices after loading state."""
        self.emotion_index.clear()
        self.temporal_index.clear()
        self.context_index.clear()
        
        # Index all memories
        for memory_list in [self.short_term_memory, self.long_term_memory, self.significant_memory]:
            for experience in memory_list:
                self._update_indices(experience)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the current memory state."""
        return {
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory),
            'significant_count': len(self.significant_memory),
            'total_memories': (
                len(self.short_term_memory) +
                len(self.long_term_memory) +
                len(self.significant_memory)
            ),
            'emotion_distribution': self._get_emotion_distribution(),
            'average_strength': self._calculate_average_strength()
        }

    def _get_emotion_distribution(self) -> Dict[str, float]:
        """Calculate distribution of emotions across all memories."""
        emotion_counts = {}
        total_memories = 0
        
        for memory_list in [self.short_term_memory, self.long_term_memory, self.significant_memory]:
            for experience in memory_list:
                total_memories += 1
                for emotion, intensity in experience.emotional_state.items():
                    if intensity > 0.5:  # Count only significant emotions
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Convert to percentages
        if total_memories > 0:
            return {
                emotion: (count / total_memories) * 100
                for emotion, count in emotion_counts.items()
            }
        return {}

    def _calculate_average_strength(self) -> float:
        """Calculate average memory strength across all memories."""
        total_strength = 0
        total_memories = 0
        
        for memory_list in [self.short_term_memory, self.long_term_memory, self.significant_memory]:
            for experience in memory_list:
                total_strength += experience.strength
                total_memories += 1
        
        return total_strength / total_memories if total_memories > 0 else 0.0

    def get_emotional_trajectory(
        self,
        experience_id: str,
        window_size: int = 5
    ) -> List[Dict[str, float]]:
        """Get emotional trajectory leading to a specific experience."""
        # Find target experience
        target_exp = None
        for memory_list in [self.short_term_memory, self.long_term_memory, self.significant_memory]:
            for exp in memory_list:
                if exp.id == experience_id:
                    target_exp = exp
                    break
            if target_exp:
                break
        
        if not target_exp:
            return []
        
        # Get temporal context
        temporal_window = []
        for timestamp, exp in sorted(self.temporal_index):
            if exp.timestamp < target_exp.timestamp:
                temporal_window.append({
                    'valence': exp.valence,
                    'arousal': exp.arousal,
                    'dominance': exp.dominance,
                    'timestamp': exp.timestamp
                })
                if len(temporal_window) > window_size:
                    temporal_window.pop(0)
        
        return temporal_window

# SELAgents API Documentation

## Core Components

### Agent
```python
class Agent:
    """Base agent class incorporating emotional and social capabilities."""
    
    def __init__(
        self,
        agent_id: str,
        emotional_capacity: float = 0.8,
        emotion_engine: Optional[EmotionEngine] = None,
        theory_of_mind: Optional[TheoryOfMind] = None,
        emotional_memory: Optional[EmotionalMemory] = None,
        social_strategies: Optional[SocialStrategies] = None
    )
```

**Parameters:**
- `agent_id`: Unique identifier for the agent
- `emotional_capacity`: Base emotional processing capability (0.0 to 1.0)
- `emotion_engine`: Optional emotion processing system
- `theory_of_mind`: Optional social reasoning component
- `emotional_memory`: Optional emotional learning system
- `social_strategies`: Optional strategic decision-making component

**Methods:**
- `interact(other_agent: Agent) -> Dict`: Interact with another agent
- `get_emotional_state() -> np.ndarray`: Get current emotional state
- `update_strategy(context: Dict) -> None`: Update behavioral strategy

### EmotionEngine
```python
class EmotionEngine:
    """Emotional processing system for valuation and response."""
    
    def process_emotion(
        self,
        stimulus: np.ndarray,
        context: Dict
    ) -> np.ndarray
```

**Methods:**
- `process_emotion()`: Process emotional stimuli
- `update_state()`: Update emotional state
- `get_valuation()`: Get emotional valuation of situation

### SocialNetwork
```python
class SocialNetwork:
    """Manages social network dynamics and relationships."""
    
    def add_agent(self, agent: Agent) -> None
    def update_connection(
        self,
        agent1_id: str,
        agent2_id: str,
        weight: float
    ) -> None
```

## Models

### TheoryOfMind
```python
class TheoryOfMind:
    """Models other agents' mental states and intentions."""
    
    def analyze_social_context(
        self,
        network: nx.Graph
    ) -> Dict[str, float]
```

### EmotionalMemory
```python
class EmotionalMemory:
    """Stores and retrieves emotional experiences."""
    
    def store_experience(
        self,
        experience: Dict
    ) -> None
```

### SocialStrategies
```python
class SocialStrategies:
    """Implements game theory-based social strategies."""
    
    def select_strategy(
        self,
        context: Dict,
        history: List[Dict]
    ) -> Strategy
```

## Utilities

### MetricsCalculator
```python
class MetricsCalculator:
    """Calculates performance and behavioral metrics."""
    
    def calculate_emotional_intelligence(
        self,
        predicted: np.ndarray,
        actual: np.ndarray
    ) -> Dict[str, float]
```

### Visualizer
```python
class Visualizer:
    """Creates visualizations of agent interactions."""
    
    def plot_emotional_dynamics(
        self,
        emotional_states: List[Dict]
    ) -> None
```

For complete method signatures and detailed documentation, see the individual class documentation pages.

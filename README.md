# SELAgents
SELAgents: Social-Emotional Learning Framework for Artificial Agents

A comprehensive framework for implementing social and emotional learning in artificial agents, combining reinforcement learning with emotional processing models and theory of mind capabilities.

## Features

- **Emotional Processing Engine**: Advanced emotion modeling and response system
- **Theory of Mind Implementation**: Social reasoning and prediction capabilities
- **Social Strategy Framework**: Game theory-based decision making
- **Dynamic Social Networks**: Complex social relationship modeling
- **Emotional Memory System**: Learning from past social interactions
- **Visualization Tools**: Network and emotional dynamics visualization

## Installation

```bash
pip install selagents
```

## Quick Start

```python
from selagents.core import Agent
from selagents.models import TheoryOfMind
from selagents.utils import Visualizer

# Create agents with emotional capabilities
agent1 = Agent(emotional_capacity=0.8)
agent2 = Agent(emotional_capacity=0.7)

# Initialize social interaction
interaction = agent1.interact(agent2)

# Visualize the emotional dynamics
visualizer = Visualizer()
visualizer.plot_emotional_dynamics(interaction)
```

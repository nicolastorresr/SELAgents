# SELAgents Tutorial

## Quick Start Guide

### 1. Basic Agent Creation and Interaction

```python
from selagents import Agent, EmotionEngine

# Create two agents
agent1 = Agent(agent_id="agent_1", emotional_capacity=0.8)
agent2 = Agent(agent_id="agent_2", emotional_capacity=0.7)

# Run a simple interaction
interaction_result = agent1.interact(agent2)
print(f"Interaction outcome: {interaction_result['outcome']}")
```

### 2. Setting Up a Social Network

```python
from selagents import SocialNetwork

# Create network and add agents
network = SocialNetwork()
network.add_agent(agent1)
network.add_agent(agent2)

# Update connection based on interaction
network.update_connection(
    agent1.agent_id,
    agent2.agent_id,
    interaction_result['outcome']
)
```

### 3. Analyzing Emotional Dynamics

```python
from selagents.utils import MetricsCalculator, Visualizer

# Calculate metrics
metrics = MetricsCalculator()
emotional_metrics = metrics.calculate_emotional_intelligence(
    agent1.get_emotional_state(),
    agent2.get_emotional_state()
)

# Visualize results
visualizer = Visualizer()
visualizer.plot_emotional_dynamics([
    {'agent_id': agent1.agent_id, 'state': agent1.get_emotional_state()},
    {'agent_id': agent2.agent_id, 'state': agent2.get_emotional_state()}
])
```

## Advanced Usage

### 1. Implementing Theory of Mind

```python
from selagents import TheoryOfMind

# Create agent with theory of mind
agent = Agent(
    agent_id="advanced_agent",
    theory_of_mind=TheoryOfMind()
)

# Analyze social context
social_understanding = agent.theory_of_mind.analyze_social_context(
    network.get_local_network(agent.agent_id)
)
```

### 2. Coalition Formation

```python
from selagents import SocialStrategies

# Create agents with strategic capabilities
strategic_agent = Agent(
    agent_id="strategic_agent",
    social_strategies=SocialStrategies()
)

# Select and execute strategy
strategy = strategic_agent.social_strategies.select_strategy(
    context={'coalition_threshold': 0.7},
    history=interaction_history
)
```

### 3. Complex Scenarios

See the full example in `examples/complex_scenario.py` for:
- Dynamic environment handling
- Resource management
- Coalition formation and management
- Advanced metrics tracking

## Best Practices

1. **Agent Configuration**
   - Set appropriate emotional capacities based on desired behavior
   - Initialize all components needed for your use case
   - Use consistent agent IDs for tracking

2. **Network Management**
   - Update connections after each interaction
   - Monitor network metrics for stability
   - Handle coalition formation carefully

3. **Metrics and Visualization**
   - Track metrics consistently throughout simulation
   - Use appropriate visualizations for your analysis
   - Export metrics for further analysis

## Common Patterns

### 1. Environment Setup
```python
def setup_environment(num_agents: int):
    network = SocialNetwork()
    agents = []
    
    for i in range(num_agents):
        agent = Agent(
            agent_id=f"agent_{i}",
            emotional_capacity=np.random.uniform(0.5, 1.0)
        )
        agents.append(agent)
        network.add_agent(agent)
    
    return network, agents
```

### 2. Interaction Loop
```python
def run_interactions(agents, steps: int):
    history = []
    
    for step in range(steps):
        for agent in agents:
            partner = random.choice([a for a in agents if a != agent])
            interaction = agent.interact(partner)
            history.append(interaction)
    
    return history
```

### 3. Analysis Pattern
```python
def analyze_results(history, network):
    metrics = MetricsCalculator()
    
    results = {
        'interaction': metrics.calculate_interaction_metrics(history),
        'social': metrics.calculate_social_metrics(network.graph),
        'learning': metrics.calculate_learning_metrics(history)
    }
    
    return results
```

## Further Reading

- Check the API documentation for detailed method signatures
- See `examples/` directory for complete working examples
- Visit our GitHub repository for latest updates and contributions

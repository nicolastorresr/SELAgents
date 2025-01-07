"""
Basic simulation example demonstrating core functionality of SELAgents framework.
Shows how to create agents, enable emotional processing, and run simple social interactions.
"""

import numpy as np
import matplotlib.pyplot as plt
from selagents import Agent, EmotionEngine, SocialNetwork
from selagents.utils import MetricsCalculator, Visualizer

def run_basic_simulation(num_agents=5, num_steps=100):
    """
    Run a basic simulation with emotional agents interacting in a social network.
    
    Args:
        num_agents: Number of agents to create
        num_steps: Number of simulation steps to run
    """
    # Initialize components
    social_network = SocialNetwork()
    metrics = MetricsCalculator()
    visualizer = Visualizer()
    
    # Create agents with different emotional capacities
    agents = []
    for i in range(num_agents):
        emotional_capacity = np.random.uniform(0.5, 1.0)
        agent = Agent(
            agent_id=f"agent_{i}",
            emotional_capacity=emotional_capacity,
            emotion_engine=EmotionEngine()
        )
        agents.append(agent)
        social_network.add_agent(agent)
    
    # Run simulation
    interaction_history = []
    emotional_states = []
    
    for step in range(num_steps):
        # Each agent interacts with a random other agent
        for agent in agents:
            # Select interaction partner
            other_agents = [a for a in agents if a != agent]
            partner = np.random.choice(other_agents)
            
            # Perform interaction
            interaction = agent.interact(partner)
            interaction_history.append(interaction)
            
            # Update emotional states
            emotional_states.append({
                'step': step,
                'agent_id': agent.agent_id,
                'emotional_state': agent.get_emotional_state()
            })
            
            # Update social network
            social_network.update_connection(
                agent.agent_id,
                partner.agent_id,
                interaction['outcome']
            )
    
    # Calculate metrics
    final_metrics = {
        'interaction': metrics.calculate_interaction_metrics(interaction_history),
        'emotional': metrics.calculate_emotional_dynamics(emotional_states),
        'social': metrics.calculate_social_metrics(social_network.graph)
    }
    
    # Visualize results
    visualizer.plot_emotional_dynamics(emotional_states)
    visualizer.plot_social_network(social_network.graph)
    
    return final_metrics

def main():
    """Main function to run the basic simulation."""
    print("Starting basic SELAgents simulation...")
    
    # Run simulation
    metrics = run_basic_simulation()
    
    # Print results
    print("\nSimulation Results:")
    print("-" * 50)
    for metric_type, values in metrics.items():
        print(f"\n{metric_type.title()} Metrics:")
        for name, value in values.items():
            print(f"{name}: {value:.3f}")
    
    print("\nVisualization windows should now be open.")
    plt.show()

if __name__ == "__main__":
    main()

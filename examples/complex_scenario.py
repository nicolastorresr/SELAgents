"""
Complex scenario example demonstrating advanced features of SELAgents framework.
Includes theory of mind, emotional memory, and social strategies in a dynamic environment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from selagents import (
    Agent, EmotionEngine, SocialNetwork, 
    TheoryOfMind, EmotionalMemory, SocialStrategies
)
from selagents.utils import MetricsCalculator, Visualizer

class ComplexScenario:
    """Class to manage a complex social-emotional learning scenario."""
    
    def __init__(
        self,
        num_agents=10,
        environment_volatility=0.3,
        coalition_threshold=0.7
    ):
        """
        Initialize complex scenario.
        
        Args:
            num_agents: Number of agents to create
            environment_volatility: Rate of environmental change
            coalition_threshold: Threshold for coalition formation
        """
        self.num_agents = num_agents
        self.environment_volatility = environment_volatility
        self.coalition_threshold = coalition_threshold
        
        # Initialize components
        self.social_network = SocialNetwork()
        self.metrics = MetricsCalculator()
        self.visualizer = Visualizer()
        
        # Create agents with advanced capabilities
        self.agents = self._initialize_agents()
        
        # Track coalitions and resources
        self.coalitions = {}
        self.resources = self._initialize_resources()
    
    def _initialize_agents(self):
        """Initialize agents with advanced cognitive and social capabilities."""
        agents = []
        
        for i in range(self.num_agents):
            # Create components for each agent
            emotion_engine = EmotionEngine()
            theory_of_mind = TheoryOfMind()
            emotional_memory = EmotionalMemory()
            social_strategies = SocialStrategies()
            
            # Initialize agent with all components
            agent = Agent(
                agent_id=f"agent_{i}",
                emotional_capacity=np.random.uniform(0.6, 1.0),
                emotion_engine=emotion_engine,
                theory_of_mind=theory_of_mind,
                emotional_memory=emotional_memory,
                social_strategies=social_strategies
            )
            
            agents.append(agent)
            self.social_network.add_agent(agent)
            
        return agents
    
    def _initialize_resources(self):
        """Initialize resource distribution in the environment."""
        return {
            f"resource_{i}": np.random.uniform(0.5, 1.0)
            for i in range(self.num_agents // 2)
        }
    
    def _update_environment(self):
        """Update environmental conditions and resource availability."""
        if np.random.random() < self.environment_volatility:
            # Modify existing resources
            for resource in self.resources:
                self.resources[resource] *= np.random.uniform(0.8, 1.2)
                
            # Potentially add new resource
            if np.random.random() < 0.3:
                new_resource = f"resource_{len(self.resources)}"
                self.resources[new_resource] = np.random.uniform(0.5, 1.0)
    
    def run_simulation(self, num_steps=200):
        """
        Run the complex simulation scenario.
        
        Args:
            num_steps: Number of simulation steps to run
        
        Returns:
            Dictionary of metrics from the simulation
        """
        history = {
            'interactions': [],
            'emotions': [],
            'coalitions': [],
            'resources': []
        }
        
        for step in range(num_steps):
            # Update environment
            self._update_environment()
            
            # Agent interactions and decisions
            for agent in self.agents:
                # Analyze social situation
                social_analysis = agent.theory_of_mind.analyze_social_context(
                    self.social_network.get_local_network(agent.agent_id)
                )
                
                # Choose interaction strategy
                strategy = agent.social_strategies.select_strategy(
                    social_analysis,
                    agent.emotional_memory.get_relevant_experiences()
                )
                
                # Select interaction partner
                partner = strategy.choose_partner(
                    [a for a in self.agents if a != agent]
                )
                
                # Perform interaction
                interaction = agent.interact(
                    partner,
                    context={'resources': self.resources}
                )
                
                # Update histories
                history['interactions'].append(interaction)
                history['emotions'].append({
                    'step': step,
                    'agent_id': agent.agent_id,
                    'emotional_state': agent.get_emotional_state()
                })
                
                # Update social network
                self.social_network.update_connection(
                    agent.agent_id,
                    partner.agent_id,
                    interaction['outcome']
                )
                
                # Consider coalition formation
                if interaction['outcome'] > self.coalition_threshold:
                    self._handle_coalition_formation(agent, partner)
            
            # Record coalition and resource states
            history['coalitions'].append(self.coalitions.copy())
            history['resources'].append(self.resources.copy())
            
            # Visualize current state periodically
            if step % 50 == 0:
                self.visualizer.plot_current_state(
                    self.social_network.graph,
                    self.coalitions,
                    self.resources
                )
        
        return self._calculate_final_metrics(history)
    
    def _handle_coalition_formation(self, agent1, agent2):
        """Handle potential coalition formation between agents."""
        # Check if either agent is already in a coalition
        coalition1 = None
        coalition2 = None
        
        for coalition_id, members in self.coalitions.items():
            if agent1.agent_id in members:
                coalition1 = coalition_id
            if agent2.agent_id in members:
                coalition2 = coalition_id
        
        # Handle coalition formation/merging
        if coalition1 is None and coalition2 is None:
            # Form new coalition
            coalition_id = f"coalition_{len(self.coalitions)}"
            self.coalitions[coalition_id] = {
                agent1.agent_id,
                agent2.agent_id
            }
        elif coalition1 is None:
            # Add agent1 to agent2's coalition
            self.coalitions[coalition2].add(agent1.agent_id)
        elif coalition2 is None:
            # Add agent2 to agent1's coalition
            self.coalitions[coalition1].add(agent2.agent_id)
        elif coalition1 != coalition2:
            # Merge coalitions
            self.coalitions[coalition1].update(
                self.coalitions[coalition2]
            )
            del self.coalitions[coalition2]
    
    def _calculate_final_metrics(self, history):
        """Calculate comprehensive metrics from simulation history."""
        return {
            'interaction': self.metrics.calculate_interaction_metrics(
                history['interactions']
            ),
            'emotional': self.metrics.calculate_emotional_dynamics(
                pd.DataFrame(history['emotions'])
            ),
            'social': self.metrics.calculate_social_metrics(
                self.social_network.graph
            ),
            'coalition': self.metrics.calculate_coalition_metrics({
                'history': history['coalitions'],
                'final_state': self.coalitions
            }),
            'learning': self.metrics.calculate_learning_metrics(
                history['interactions']
            )
        }

def main():
    """Main function to run the complex scenario."""
    print("Starting complex SELAgents scenario...")
    
    # Create and run scenario
    scenario = ComplexScenario()
    metrics = scenario.run_simulation()
    
    # Print results
    print("\nScenario Results:")
    print("-" * 50)
    for metric_type, values in metrics.items():
        print(f"\n{metric_type.title()} Metrics:")
        for name, value in values.items():
            print(f"{name}: {value:.3f}")
    
    print("\nVisualization windows should now be open.")
    plt.show()

if __name__ == "__main__":
    main()

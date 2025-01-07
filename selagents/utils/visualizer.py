"""
selagents/utils/visualizer.py

Visualization tools for social networks, emotional dynamics, and agent interactions.
Creates plots and animations of agent behaviors and network structures.
"""

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import pandas as pd
from matplotlib.animation import FuncAnimation
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Class for creating visualizations of agent interactions and dynamics."""
    
    def __init__(self, style: str = 'whitegrid'):
        """
        Initialize visualizer with specified style.
        
        Args:
            style: seaborn style name ('whitegrid', 'darkgrid', etc.)
        """
        self.style = style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def plot_social_network(
        self,
        network: nx.Graph,
        node_colors: Optional[Dict[str, str]] = None,
        node_sizes: Optional[Dict[str, float]] = None,
        title: str = "Agent Social Network",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot social network structure with customizable node properties.
        
        Args:
            network: NetworkX graph of social connections
            node_colors: Dict mapping node IDs to colors
            node_sizes: Dict mapping node IDs to sizes
            title: Plot title
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(15, 10))
        
        # Set default node properties if not provided
        if not node_colors:
            node_colors = {node: '#1f77b4' for node in network.nodes()}
        if not node_sizes:
            node_sizes = {node: 1000 for node in network.nodes()}
            
        # Calculate edge weights for thickness
        edge_weights = [
            network[u][v].get('weight', 1.0) * 2
            for u, v in network.edges()
        ]
        
        # Create layout
        pos = nx.spring_layout(network)
        
        # Draw network
        nx.draw_networkx_nodes(
            network, pos,
            node_color=[node_colors[node] for node in network.nodes()],
            node_size=[node_sizes[node] for node in network.nodes()]
        )
        
        nx.draw_networkx_edges(
            network, pos,
            width=edge_weights,
            alpha=0.5
        )
        
        nx.draw_networkx_labels(network, pos)
        
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved network visualization to {save_path}")
            
        return plt.gcf()
    
    def plot_emotional_dynamics(
        self,
        emotional_data: pd.DataFrame,
        agents: Optional[List[str]] = None,
        emotions: Optional[List[str]] = None,
        title: str = "Emotional Dynamics Over Time",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot emotional dynamics over time for multiple agents.
        
        Args:
            emotional_data: DataFrame with columns [timestamp, agent_id, emotion, value]
            agents: List of agent IDs to include
            emotions: List of emotions to plot
            title: Plot title
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(15, 8))
        
        # Filter data if needed
        plot_data = emotional_data.copy()
        if agents:
            plot_data = plot_data[plot_data['agent_id'].isin(agents)]
        if emotions:
            plot_data = plot_data[plot_data['emotion'].isin(emotions)]
            
        # Create plot
        sns.lineplot(
            data=plot_data,
            x='timestamp',
            y='value',
            hue='agent_id',
            style='emotion',
            markers=True,
            dashes=False
        )
        
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Emotional Intensity")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved emotional dynamics plot to {save_path}")
            
        return plt.gcf()
    
    def plot_coalition_dynamics(
        self,
        coalition_data: pd.DataFrame,
        metrics: List[str] = ['stability', 'trust', 'resource_efficiency'],
        title: str = "Coalition Dynamics",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot coalition metrics over time.
        
        Args:
            coalition_data: DataFrame with coalition metrics over time
            metrics: List of metrics to plot
            title: Plot title
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(15, 8))
        
        for metric in metrics:
            plt.plot(
                coalition_data['timestamp'],
                coalition_data[metric],
                label=metric.replace('_', ' ').title(),
                marker='o'
            )
            
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Metric Value")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved coalition dynamics plot to {save_path}")
            
        return plt.gcf()
    
    def create_interaction_animation(
        self,
        interaction_sequence: List[Dict],
        network: nx.Graph,
        interval: int = 500,
        save_path: Optional[str] = None
    ) -> FuncAnimation:
        """
        Create animation of agent interactions over time.
        
        Args:
            interaction_sequence: List of interaction states
            network: Social network structure
            interval: Animation interval in milliseconds
            save_path: Optional path to save animation
        """
        fig, ax = plt.subplots(figsize=(15, 10))
        
        def update(frame):
            ax.clear()
            state = interaction_sequence[frame]
            
            # Update network visualization
            pos = nx.spring_layout(network)
            nx.draw(
                network, pos,
                node_color=[state['node_colors'].get(node, '#1f77b4') for node in network.nodes()],
                node_size=[state['node_sizes'].get(node, 1000) for node in network.nodes()],
                ax=ax
            )
            
            ax.set_title(f"Time: {frame}")
            
        anim = FuncAnimation(
            fig,
            update,
            frames=len(interaction_sequence),
            interval=interval,
            blit=False
        )
        
        if save_path:
            anim.save(save_path)
            logger.info(f"Saved interaction animation to {save_path}")
            
        return anim
    
    def plot_emotional_heatmap(
        self,
        emotional_matrix: np.ndarray,
        agent_labels: List[str],
        emotion_labels: List[str],
        title: str = "Emotional State Heatmap",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create heatmap of emotional states across agents.
        
        Args:
            emotional_matrix: 2D array of emotional values
            agent_labels: List of agent IDs
            emotion_labels: List of emotion names
            title: Plot title
            save_path: Optional path to save figure
        """
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(
            emotional_matrix,
            xticklabels=emotion_labels,
            yticklabels=agent_labels,
            cmap='RdYlBu_r',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Emotional Intensity'}
        )
        
        plt.title(title)
        plt.xlabel("Emotions")
        plt.ylabel("Agents")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved emotional heatmap to {save_path}")
            
        return plt.gcf()

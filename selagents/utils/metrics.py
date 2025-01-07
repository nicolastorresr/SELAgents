"""
selagents/utils/metrics.py

Evaluation metrics for measuring agent performance, social dynamics,
and emotional intelligence in social-emotional learning systems.
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Dict, List, Optional, Union, Tuple
import networkx as nx
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Class for calculating various performance and behavioral metrics."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.metric_history: Dict[str, List[float]] = {}
        
    def calculate_emotional_intelligence(
        self,
        predicted_emotions: np.ndarray,
        actual_emotions: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate emotional intelligence metrics.
        
        Args:
            predicted_emotions: Agent's emotion predictions
            actual_emotions: Actual emotional states
            
        Returns:
            Dict of emotional intelligence metrics
        """
        metrics_dict = {}
        
        # Calculate accuracy
        metrics_dict['emotion_accuracy'] = metrics.accuracy_score(
            actual_emotions.argmax(axis=1),
            predicted_emotions.argmax(axis=1)
        )
        
        # Calculate precision, recall, f1 per emotion
        metrics_dict['emotion_precision'] = metrics.precision_score(
            actual_emotions.argmax(axis=1),
            predicted_emotions.argmax(axis=1),
            average='weighted'
        )
        
        metrics_dict['emotion_recall'] = metrics.recall_score(
            actual_emotions.argmax(axis=1),
            predicted_emotions.argmax(axis=1),
            average='weighted'
        )
        
        metrics_dict['emotion_f1'] = metrics.f1_score(
            actual_emotions.argmax(axis=1),
            predicted_emotions.argmax(axis=1),
            average='weighted'
        )
        
        return metrics_dict
    
    def calculate_social_metrics(
        self,
        network: nx.Graph,
        agent_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate social network metrics for an agent or entire network.
        
        Args:
            network: Social network graph
            agent_id: Optional specific agent to analyze
            
        Returns:
            Dict of social metrics
        """
        metrics_dict = {}
        
        if agent_id:
            # Individual agent metrics
            metrics_dict['degree_centrality'] = nx.degree_centrality(network)[agent_id]
            metrics_dict['betweenness_centrality'] = nx.betweenness_centrality(network)[agent_id]
            metrics_dict['closeness_centrality'] = nx.closeness_centrality(network)[agent_id]
            
            # Calculate local clustering
            metrics_dict['clustering_coefficient'] = nx.clustering(network, agent_id)
            
            # Calculate average neighbor degree
            metrics_dict['avg_neighbor_degree'] = np.mean([
                network.degree(neighbor)
                for neighbor in network.neighbors(agent_id)
            ]) if list(network.neighbors(agent_id)) else 0
            
        else:
            # Global network metrics
            metrics_dict['average_clustering'] = nx.average_clustering(network)
            metrics_dict['network_density'] = nx.density(network)
            metrics_dict['average_shortest_path'] = nx.average_shortest_path_length(network)
            metrics_dict['network_diameter'] = nx.diameter(network)
            
            # Calculate degree distribution entropy
            degrees = [d for n, d in network.degree()]
            metrics_dict['degree_distribution_entropy'] = entropy(
                np.bincount(degrees) / len(degrees)
            )
            
        return metrics_dict
    
    def calculate_coalition_metrics(
        self,
        coalition_data: Dict
    ) -> Dict[str, float]:
        """
        Calculate metrics for coalition performance.
        
        Args:
            coalition_data: Dictionary containing coalition information
            
        Returns:
            Dict of coalition metrics
        """
        metrics_dict = {}
        
        # Calculate stability metrics
        metrics_dict['member_retention'] = len(coalition_data['current_members']) / \
                                         len(coalition_data['initial_members'])
        
        metrics_dict['resource_efficiency'] = coalition_data['shared_resources'] / \
                                            (len(coalition_data['current_members']) * 100)
        
        # Calculate trust metrics
        trust_matrix = np.array(list(coalition_data['trust_matrix'].values()))
        metrics_dict['average_trust'] = np.mean(trust_matrix)
        metrics_dict['trust_variance'] = np.var(trust_matrix)
        
        # Calculate hierarchy metrics
        hierarchy_positions = list(coalition_data['hierarchy'].values())
        metrics_dict['hierarchy_entropy'] = entropy(
            np.bincount(hierarchy_positions) / len(hierarchy_positions)
        )
        
        return metrics_dict
    
    def calculate_interaction_metrics(
        self,
        interaction_history: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate metrics for interaction patterns.
        
        Args:
            interaction_history: List of interaction records
            
        Returns:
            Dict of interaction metrics
        """
        metrics_dict = {}
        
        # Calculate interaction frequency
        metrics_dict['interaction_frequency'] = len(interaction_history) / \
                                              (interaction_history[-1]['timestamp'] - 
                                               interaction_history[0]['timestamp'])
        
        # Calculate success rate
        metrics_dict['interaction_success_rate'] = np.mean([
            record['outcome'] > 0.5 for record in interaction_history
        ])
        
        # Calculate reciprocity
        interaction_pairs = [(record['agent_id'], record['target_id'])
                           for record in interaction_history]
        metrics_dict['reciprocity'] = len(set(
            pair for pair in interaction_pairs
            if (pair[1], pair[0]) in interaction_pairs
        )) / len(set(interaction_pairs))
        
        return metrics_dict
    
    def calculate_emotional_dynamics(
        self,
        emotional_history: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate metrics for emotional dynamics.
        
        Args:
            emotional_history: DataFrame of emotional states over time
            
        Returns:
            Dict of emotional dynamics metrics
        """
        metrics_dict = {}
        
        # Calculate emotional volatility
        metrics_dict['emotional_volatility'] = np.mean([
            np.std(group['value'])
            for _, group in emotional_history.groupby('agent_id')
        ])
        
        # Calculate emotional convergence
        final_states = emotional_history.groupby('agent_id').last()['value']
        metrics_dict['emotional_convergence'] = np.std(final_states)
        
        # Calculate emotional contagion
        lag_correlation = emotional_history.groupby('agent_id')['value'].apply(
            lambda x: x.autocorr(lag=1)
        ).mean()
        metrics_dict['emotional_contagion'] = max(0, lag_correlation)
        
        return metrics_dict
    
    def calculate_learning_metrics(
            self,
            learning_history: List[Dict]
        ) -> Dict[str, float]:
            """
            Calculate metrics related to agent learning and adaptation.
            
            Args:
                learning_history: List of learning event records
                
            Returns:
                Dict of learning metrics
            """
            metrics_dict = {}
            
            # Calculate learning rate
            if len(learning_history) > 1:
                performance_values = [record['performance'] for record in learning_history]
                metrics_dict['learning_rate'] = np.polyfit(
                    range(len(performance_values)),
                    performance_values,
                    1
                )[0]
            else:
                metrics_dict['learning_rate'] = 0.0
                
            # Calculate adaptation speed
            adaptation_times = [
                record['adaptation_time']
                for record in learning_history
                if 'adaptation_time' in record
            ]
            metrics_dict['adaptation_speed'] = (
                1 / np.mean(adaptation_times) if adaptation_times else 0.0
            )
            
            # Calculate strategy diversity
            strategies = [record['strategy'] for record in learning_history]
            strategy_counts = pd.Series(strategies).value_counts(normalize=True)
            metrics_dict['strategy_diversity'] = entropy(strategy_counts)
            
            # Calculate exploration vs exploitation ratio
            exploration_actions = sum(
                1 for record in learning_history
                if record.get('action_type') == 'exploration'
            )
            metrics_dict['exploration_ratio'] = (
                exploration_actions / len(learning_history)
                if learning_history else 0.0
            )
            
            return metrics_dict
        
        def aggregate_metrics(
            self,
            metric_sets: List[Dict[str, float]],
            weights: Optional[Dict[str, float]] = None
        ) -> float:
            """
            Aggregate multiple metric sets into a single performance score.
            
            Args:
                metric_sets: List of metric dictionaries to aggregate
                weights: Optional dictionary of metric weights
                
            Returns:
                Float representing overall performance score
            """
            if not weights:
                weights = {metric: 1.0 for metrics in metric_sets for metric in metrics}
                
            total_score = 0.0
            total_weight = 0.0
            
            for metrics in metric_sets:
                for metric, value in metrics.items():
                    if metric in weights:
                        total_score += value * weights[metric]
                        total_weight += weights[metric]
                        
            return total_score / total_weight if total_weight > 0 else 0.0
        
        def track_metric_history(
            self,
            metric_name: str,
            value: float
        ) -> None:
            """
            Track the history of a specific metric over time.
            
            Args:
                metric_name: Name of the metric to track
                value: Current value of the metric
            """
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []
            self.metric_history[metric_name].append(value)
            
        def get_metric_statistics(
            self,
            metric_name: str
        ) -> Dict[str, float]:
            """
            Calculate statistics for a tracked metric's history.
            
            Args:
                metric_name: Name of the metric to analyze
                
            Returns:
                Dict of statistical measures
            """
            if metric_name not in self.metric_history:
                return {}
                
            values = np.array(self.metric_history[metric_name])
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'trend': np.polyfit(range(len(values)), values, 1)[0]
            }
        
        def export_metrics(
            self,
            filepath: str,
            format: str = 'csv'
        ) -> None:
            """
            Export tracked metrics to a file.
            
            Args:
                filepath: Path to save the metrics
                format: Format to save in ('csv' or 'json')
            """
            metrics_df = pd.DataFrame(self.metric_history)
            
            if format.lower() == 'csv':
                metrics_df.to_csv(filepath, index=False)
            elif format.lower() == 'json':
                metrics_df.to_json(filepath, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Metrics exported to {filepath}")
            
        def reset_history(self) -> None:
            """Reset the metric history."""
            self.metric_history = {}

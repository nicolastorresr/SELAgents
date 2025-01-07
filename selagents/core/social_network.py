"""
core/social_network.py: Social Network Dynamics Implementation

This module manages the social network structure and dynamics between agents,
including relationship formation, trust dynamics, influence propagation,
and social group dynamics.

Key features:
- Dynamic social graph management
- Relationship strength and trust modeling
- Social influence propagation
- Group formation and community detection
- Social status and hierarchy modeling
- Network metrics and analysis

Dependencies:
    - numpy: For numerical computations
    - networkx: For graph operations
    - scipy: For mathematical operations
    - dataclasses: For structured data
"""

import numpy as np
import networkx as nx
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import random

class RelationType(Enum):
    """Types of relationships between agents."""
    FRIEND = "friend"
    ACQUAINTANCE = "acquaintance"
    NEUTRAL = "neutral"
    COMPETITOR = "competitor"
    ADVERSARY = "adversary"

@dataclass
class Relationship:
    """Represents a relationship between two agents."""
    agent_id1: str
    agent_id2: str
    relation_type: RelationType
    trust: float  # 0 to 1
    influence: float  # 0 to 1
    interaction_history: List[Dict[str, Any]]
    last_interaction: float  # timestamp
    emotional_memory: Dict[str, float]

    def __init__(
        self,
        agent_id1: str,
        agent_id2: str,
        relation_type: RelationType = RelationType.NEUTRAL,
        trust: float = 0.5,
        influence: float = 0.5
    ):
        self.agent_id1 = agent_id1
        self.agent_id2 = agent_id2
        self.relation_type = relation_type
        self.trust = trust
        self.influence = influence
        self.interaction_history = []
        self.last_interaction = 0.0
        self.emotional_memory = {}

class SocialNetwork:
    """
    Main class for managing social network dynamics between agents.
    """
    def __init__(
        self,
        trust_decay_rate: float = 0.1,
        influence_threshold: float = 0.3,
        memory_size: int = 100
    ):
        self.network = nx.Graph()
        self.trust_decay_rate = trust_decay_rate
        self.influence_threshold = influence_threshold
        self.memory_size = memory_size
        
        # Community detection cache
        self._community_cache = None
        self._last_community_update = 0
        
        # Initialize metrics tracking
        self._init_metrics_tracking()

    def _init_metrics_tracking(self) -> None:
        """Initialize network metrics tracking."""
        self.metrics_history = {
            'density': [],
            'clustering': [],
            'avg_path_length': [],
            'community_modularity': []
        }

    def add_agent(
        self,
        agent_id: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an agent to the social network."""
        if attributes is None:
            attributes = {}
        
        self.network.add_node(
            agent_id,
            attributes=attributes,
            status=0.5,  # Initial social status
            influence_radius=0.5
        )

    def add_relationship(
        self,
        agent_id1: str,
        agent_id2: str,
        relation_type: RelationType = RelationType.NEUTRAL,
        initial_trust: float = 0.5
    ) -> None:
        """Create a relationship between two agents."""
        if agent_id1 not in self.network or agent_id2 not in self.network:
            raise ValueError("Both agents must be in the network")
        
        relationship = Relationship(
            agent_id1=agent_id1,
            agent_id2=agent_id2,
            relation_type=relation_type,
            trust=initial_trust
        )
        
        self.network.add_edge(
            agent_id1,
            agent_id2,
            relationship=relationship
        )

    def update_relationship(
        self,
        agent_id1: str,
        agent_id2: str,
        interaction_result: Dict[str, Any]
    ) -> None:
        """Update relationship based on interaction outcome."""
        if not self.network.has_edge(agent_id1, agent_id2):
            return
        
        relationship = self.network[agent_id1][agent_id2]['relationship']
        
        # Update trust based on interaction outcome
        trust_delta = self._calculate_trust_update(interaction_result)
        relationship.trust = np.clip(
            relationship.trust + trust_delta,
            0,
            1
        )
        
        # Update relationship type based on trust
        relationship.relation_type = self._determine_relation_type(
            relationship.trust
        )
        
        # Update influence based on interaction success
        influence_delta = interaction_result.get('success', 0.0) * 0.1
        relationship.influence = np.clip(
            relationship.influence + influence_delta,
            0,
            1
        )
        
        # Update interaction history
        relationship.interaction_history.append({
            'timestamp': interaction_result.get('timestamp', 0.0),
            'type': interaction_result.get('type', 'unknown'),
            'success': interaction_result.get('success', 0.0),
            'emotional_impact': interaction_result.get('emotional_impact', 0.0)
        })
        
        # Maintain limited history
        if len(relationship.interaction_history) > self.memory_size:
            relationship.interaction_history.pop(0)
        
        # Update emotional memory
        emotion = interaction_result.get('emotion', 'neutral')
        relationship.emotional_memory[emotion] = relationship.emotional_memory.get(
            emotion, 0.0
        ) + interaction_result.get('emotional_intensity', 0.0)

    def _calculate_trust_update(
        self,
        interaction_result: Dict[str, Any]
    ) -> float:
        """Calculate trust update based on interaction outcome."""
        base_impact = interaction_result.get('success', 0.0) * 0.2
        
        # Consider emotional impact
        emotional_impact = interaction_result.get('emotional_impact', 0.0)
        emotional_factor = 0.1 * np.sign(emotional_impact) * abs(emotional_impact) ** 0.5
        
        # Consider interaction type
        interaction_weight = {
            'cooperation': 1.2,
            'competition': 0.8,
            'conflict': 0.5,
            'support': 1.5
        }.get(interaction_result.get('type', 'neutral'), 1.0)
        
        return (base_impact + emotional_factor) * interaction_weight

    def _determine_relation_type(self, trust: float) -> RelationType:
        """Determine relationship type based on trust level."""
        if trust >= 0.8:
            return RelationType.FRIEND
        elif trust >= 0.6:
            return RelationType.ACQUAINTANCE
        elif trust >= 0.4:
            return RelationType.NEUTRAL
        elif trust >= 0.2:
            return RelationType.COMPETITOR
        else:
            return RelationType.ADVERSARY

    def propagate_influence(
        self,
        source_id: str,
        influence_type: str,
        strength: float
    ) -> Dict[str, float]:
        """Propagate influence through the network."""
        if source_id not in self.network:
            return {}
        
        influence_map = {source_id: strength}
        visited = {source_id}
        queue = [(source_id, strength)]
        
        while queue:
            current_id, current_strength = queue.pop(0)
            
            if current_strength < self.influence_threshold:
                continue
            
            for neighbor in self.network.neighbors(current_id):
                if neighbor in visited:
                    continue
                
                relationship = self.network[current_id][neighbor]['relationship']
                propagated_strength = (
                    current_strength *
                    relationship.influence *
                    self._calculate_influence_factor(influence_type, relationship)
                )
                
                if propagated_strength >= self.influence_threshold:
                    influence_map[neighbor] = propagated_strength
                    visited.add(neighbor)
                    queue.append((neighbor, propagated_strength))
        
        return influence_map

    def _calculate_influence_factor(
        self,
        influence_type: str,
        relationship: Relationship
    ) -> float:
        """Calculate influence factor based on relationship and type."""
        base_factor = {
            'emotional': 0.8,
            'behavioral': 0.6,
            'opinion': 0.4,
            'action': 0.5
        }.get(influence_type, 0.5)
        
        relationship_factor = {
            RelationType.FRIEND: 1.2,
            RelationType.ACQUAINTANCE: 1.0,
            RelationType.NEUTRAL: 0.8,
            RelationType.COMPETITOR: 0.4,
            RelationType.ADVERSARY: 0.2
        }[relationship.relation_type]
        
        return base_factor * relationship_factor

    def detect_communities(
        self,
        force_update: bool = False
    ) -> Dict[str, Set[str]]:
        """Detect communities in the network."""
        current_time = random.random()  # Simulated timestamp
        
        if (not force_update and
            self._community_cache is not None and
            current_time - self._last_community_update < 1.0):
            return self._community_cache
        
        # Convert relationship weights to community detection weights
        weights = {
            (e[0], e[1]): self._calculate_community_weight(e[2]['relationship'])
            for e in self.network.edges(data=True)
        }
        
        # Create weighted graph for community detection
        weighted_graph = self.network.copy()
        nx.set_edge_attributes(weighted_graph, weights, 'weight')
        
        # Detect communities using Louvain method
        communities = nx.community.louvain_communities(weighted_graph)
        
        # Convert to dictionary format
        community_dict = {
            f"community_{i}": set(members)
            for i, members in enumerate(communities)
        }
        
        self._community_cache = community_dict
        self._last_community_update = current_time
        
        return community_dict

    def _calculate_community_weight(self, relationship: Relationship) -> float:
        """Calculate weight for community detection."""
        relation_weights = {
            RelationType.FRIEND: 1.0,
            RelationType.ACQUAINTANCE: 0.7,
            RelationType.NEUTRAL: 0.4,
            RelationType.COMPETITOR: 0.2,
            RelationType.ADVERSARY: 0.1
        }
        
        base_weight = relation_weights[relationship.relation_type]
        trust_factor = relationship.trust
        interaction_factor = min(1.0, len(relationship.interaction_history) / 10)
        
        return base_weight * (0.7 * trust_factor + 0.3 * interaction_factor)

    def update_social_status(self) -> None:
        """Update social status of all agents."""
        # Calculate pagerank for base status
        pagerank = nx.pagerank(
            self.network,
            weight=lambda u, v, d: d['relationship'].influence
        )
        
        # Calculate weighted degree centrality
        degree_centrality = nx.degree_centrality(self.network)
        
        # Update status for each agent
        for agent_id in self.network.nodes():
            # Combine metrics with weights
            status = (
                0.4 * pagerank[agent_id] +
                0.3 * degree_centrality[agent_id] +
                0.3 * self._calculate_relationship_quality(agent_id)
            )
            
            self.network.nodes[agent_id]['status'] = status

    def _calculate_relationship_quality(self, agent_id: str) -> float:
        """Calculate overall relationship quality for an agent."""
        if agent_id not in self.network:
            return 0.0
        
        relationships = [
            self.network[agent_id][neighbor]['relationship']
            for neighbor in self.network.neighbors(agent_id)
        ]
        
        if not relationships:
            return 0.0
        
        # Calculate average trust and influence
        avg_trust = np.mean([r.trust for r in relationships])
        avg_influence = np.mean([r.influence for r in relationships])
        
        # Consider relationship types
        type_scores = {
            RelationType.FRIEND: 1.0,
            RelationType.ACQUAINTANCE: 0.7,
            RelationType.NEUTRAL: 0.5,
            RelationType.COMPETITOR: 0.3,
            RelationType.ADVERSARY: 0.1
        }
        avg_type_score = np.mean([
            type_scores[r.relation_type] for r in relationships
        ])
        
        return (0.4 * avg_trust + 0.3 * avg_influence + 0.3 * avg_type_score)

    def get_network_metrics(self) -> Dict[str, float]:
        """Calculate and return network metrics."""
        metrics = {
            'density': nx.density(self.network),
            'clustering': nx.average_clustering(self.network),
            'avg_path_length': nx.average_shortest_path_length(self.network)
            if nx.is_connected(self.network) else float('inf'),
            'community_modularity': self._calculate_modularity()
        }
        
        # Update metrics history
        for metric, value in metrics.items():
            self.metrics_history[metric].append(value)
            if len(self.metrics_history[metric]) > self.memory_size:
                self.metrics_history[metric].pop(0)
        
        return metrics

    def _calculate_modularity(self) -> float:
        """Calculate network modularity based on communities."""
        communities = self.detect_communities(force_update=True)
        community_sets = list(communities.values())
        
        return nx.community.modularity(
            self.network,
            community_sets,
            weight=lambda u, v, d: d['relationship'].trust
        )

    def get_agent_neighborhood(
        self,
        agent_id: str,
        depth: int = 1
    ) -> Set[str]:
        """Get agents in the neighborhood up to specified depth."""
        if agent_id not in self.network:
            return set()
        
        neighborhood = set()
        current_layer = {agent_id}
        
        for _ in range(depth):
            next_layer = set()
            for current_id in current_layer:
                neighbors = set(self.network.neighbors(current_id))
                next_layer.update(neighbors - neighborhood)
            
            neighborhood.update(next_layer)
            current_layer = next_layer
            
            if not current_layer:
                break
        
        neighborhood.remove(agent_id)
        return neighborhood

    def decay_relationships(self, time_elapsed: float) -> None:
        """Apply decay to relationships based on time elapsed."""
        for edge in self.network.edges(data=True):
            relationship = edge[2]['relationship']
            
            # Calculate time since last interaction
            decay_factor = np.exp(
                -self.trust_decay_rate *
                (time_elapsed - relationship.last_interaction)
            )
            
            # Apply decay to trust and influence
            relationship.trust *= decay_factor
            relationship.influence *= decay_factor

    def get_relationship_stats(
        self,
        agent_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics about an agent's relationships.
        
        Args:
            agent_id: ID of the agent to analyze
            
        Returns:
            Dictionary containing relationship statistics
        """
        if agent_id not in self.network:
            return {}
        
        relationships = [
            self.network[agent_id][neighbor]['relationship']
            for neighbor in self.network.neighbors(agent_id)
        ]
        
        if not relationships:
            return {
                'total_relationships': 0,
                'average_trust': 0.0,
                'average_influence': 0.0,
                'relationship_types': {},
                'interaction_frequency': 0.0,
                'emotional_profile': {},
                'social_reach': 0.0
            }
        
        # Calculate type distribution
        type_counts = {}
        for r in relationships:
            type_counts[r.relation_type] = type_counts.get(r.relation_type, 0) + 1
        
        # Calculate interaction frequency (interactions per time unit)
        total_interactions = sum(
            len(r.interaction_history) for r in relationships
        )
        time_span = max(
            1.0,
            max(r.last_interaction for r in relationships)
        )
        
        # Aggregate emotional memory across relationships
        emotional_profile = {}
        for r in relationships:
            for emotion, intensity in r.emotional_memory.items():
                emotional_profile[emotion] = emotional_profile.get(
                    emotion, 0.0
                ) + intensity
        
        # Normalize emotional profile
        total_intensity = sum(emotional_profile.values()) or 1.0
        emotional_profile = {
            k: v/total_intensity for k, v in emotional_profile.items()
        }
        
        # Calculate social reach (weighted sum of relationship influences)
        social_reach = sum(
            r.influence * r.trust for r in relationships
        ) / len(relationships)
        
        return {
            'total_relationships': len(relationships),
            'average_trust': np.mean([r.trust for r in relationships]),
            'average_influence': np.mean([r.influence for r in relationships]),
            'relationship_types': type_counts,
            'interaction_frequency': total_interactions / time_span,
            'emotional_profile': emotional_profile,
            'social_reach': social_reach
        }

    def analyze_network_evolution(
        self,
        time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze the evolution of network metrics over time.
        
        Args:
            time_window: Optional number of past measurements to analyze
            
        Returns:
            Dictionary containing evolution metrics
        """
        if time_window is None:
            time_window = len(self.metrics_history['density'])
        
        window = slice(-time_window, None)
        
        # Calculate trends for each metric
        trends = {}
        for metric, history in self.metrics_history.items():
            values = np.array(history[window])
            if len(values) > 1:
                slope, _, r_value, _, _ = stats.linregress(
                    range(len(values)),
                    values
                )
                trends[f'{metric}_trend'] = {
                    'slope': slope,
                    'r_squared': r_value ** 2
                }
            
        # Calculate stability metrics
        stability = {}
        for metric, history in self.metrics_history.items():
            values = np.array(history[window])
            if len(values) > 0:
                stability[f'{metric}_stability'] = {
                    'std': np.std(values),
                    'range': np.ptp(values)
                }
        
        return {
            'trends': trends,
            'stability': stability,
            'current_metrics': self.get_network_metrics()
        }

    def get_social_influence_paths(
        self,
        source_id: str,
        target_id: str,
        min_influence: float = 0.3
    ) -> List[List[str]]:
        """
        Find all influence paths between two agents above minimum influence threshold.
        
        Args:
            source_id: Starting agent ID
            target_id: Target agent ID
            min_influence: Minimum influence threshold for path segments
            
        Returns:
            List of valid influence paths (each path is a list of agent IDs)
        """
        if (source_id not in self.network or
            target_id not in self.network):
            return []
        
        def get_valid_neighbors(node_id: str, visited: Set[str]) -> List[str]:
            return [
                neighbor for neighbor in self.network.neighbors(node_id)
                if (neighbor not in visited and
                    self.network[node_id][neighbor]['relationship'].influence
                    >= min_influence)
            ]
        
        valid_paths = []
        queue = [(source_id, [source_id])]
        while queue:
            current_id, current_path = queue.pop(0)
            
            for neighbor in get_valid_neighbors(current_id, set(current_path)):
                new_path = current_path + [neighbor]
                
                if neighbor == target_id:
                    valid_paths.append(new_path)
                else:
                    queue.append((neighbor, new_path))
        
        return valid_paths

    def export_network_state(self) -> Dict[str, Any]:
        """
        Export the current state of the social network.
        
        Returns:
            Dictionary containing complete network state
        """
        state = {
            'nodes': {},
            'edges': [],
            'metrics': self.get_network_metrics(),
            'communities': self.detect_communities(force_update=True)
        }
        
        # Export node data
        for node in self.network.nodes(data=True):
            state['nodes'][node[0]] = {
                'attributes': node[1].get('attributes', {}),
                'status': node[1].get('status', 0.5),
                'influence_radius': node[1].get('influence_radius', 0.5)
            }
        
        # Export edge data
        for edge in self.network.edges(data=True):
            relationship = edge[2]['relationship']
            state['edges'].append({
                'agent_id1': edge[0],
                'agent_id2': edge[1],
                'relationship': {
                    'type': relationship.relation_type.value,
                    'trust': relationship.trust,
                    'influence': relationship.influence,
                    'last_interaction': relationship.last_interaction,
                    'emotional_memory': relationship.emotional_memory.copy()
                }
            })
        
        return state

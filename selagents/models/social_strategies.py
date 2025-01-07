"""
models/social_strategies.py: Game Theory Social Strategy Implementation

This module implements game theory-based social strategies for agent interaction,
managing alliance formation, coalition dynamics, and social hierarchy evolution.
It provides mechanisms for agents to make strategic social decisions considering
trust, reputation, and reciprocity.

Features:
- Coalition formation and management
- Social hierarchy dynamics
- Trust and reputation systems
- Strategic decision-making
- Resource allocation strategies
- Alliance stability analysis
"""

import numpy as np
import networkx as nx
from scipy import optimize
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import random
from collections import defaultdict
import math
import logging

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Types of social strategies an agent can employ."""
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    RECIPROCAL = "reciprocal"
    ADAPTIVE = "adaptive"
    NEUTRAL = "neutral"

@dataclass
class SocialAction:
    """Represents a social action taken by an agent."""
    agent_id: str
    target_id: str
    action_type: str
    resource_amount: float
    timestamp: float
    context: Dict[str, Any]

@dataclass
class Coalition:
    """Represents a group of cooperating agents."""
    id: str
    members: Set[str]
    formation_time: float
    trust_matrix: Dict[Tuple[str, str], float]
    shared_resources: float
    stability_score: float
    hierarchy: Dict[str, int]

class SocialStrategist:
    """Main class for managing social strategies and interactions."""
    
    def __init__(
        self,
        agent_id: str,
        initial_resources: float = 100.0,
        trust_threshold: float = 0.5,
        memory_size: int = 1000,
        learning_rate: float = 0.1
    ):
        self.agent_id = agent_id
        self.resources = initial_resources
        self.trust_threshold = trust_threshold
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        
        # Strategy state
        self.current_strategy = StrategyType.NEUTRAL
        self.action_history: List[SocialAction] = []
        self.trust_scores: Dict[str, float] = defaultdict(float)
        self.reputation_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # Coalition management
        self.current_coalition: Optional[Coalition] = None
        self.past_coalitions: List[Coalition] = []
        
        # Network analysis
        self.social_network = nx.Graph()
        self.social_network.add_node(agent_id)
        
        # Performance metrics
        self.strategy_performance: Dict[StrategyType, float] = defaultdict(float)
        self.interaction_outcomes: List[Tuple[str, float]] = []

    def choose_action(
        self,
        target_id: str,
        context: Dict[str, Any],
        available_actions: List[str]
    ) -> Tuple[str, float]:
        """Choose a strategic action based on current state and context."""
        if not available_actions:
            return "no_action", 0.0
        
        # Calculate action utilities
        action_utilities = {}
        for action in available_actions:
            utility = self._calculate_action_utility(
                action, target_id, context
            )
            action_utilities[action] = utility
        
        # Apply strategy-specific modifiers
        if self.current_strategy == StrategyType.COOPERATIVE:
            action_utilities = self._apply_cooperative_bias(action_utilities)
        elif self.current_strategy == StrategyType.COMPETITIVE:
            action_utilities = self._apply_competitive_bias(action_utilities)
        
        # Choose best action
        best_action = max(action_utilities.items(), key=lambda x: x[1])
        action_type, utility = best_action
        
        # Calculate resource allocation
        resource_amount = self._calculate_resource_allocation(
            action_type, utility, context
        )
        
        return action_type, resource_amount

    def update_strategy(self, context: Dict[str, Any]) -> None:
        """Update current strategy based on performance and context."""
        # Calculate strategy performances
        strategy_scores = {}
        for strategy in StrategyType:
            score = self._evaluate_strategy_performance(strategy)
            strategy_scores[strategy] = score
        
        # Consider context factors
        context_factors = self._analyze_context(context)
        
        # Combine scores with context
        final_scores = {}
        for strategy, base_score in strategy_scores.items():
            context_modifier = self._get_context_modifier(
                strategy, context_factors
            )
            final_scores[strategy] = base_score * context_modifier
        
        # Select best strategy
        best_strategy = max(final_scores.items(), key=lambda x: x[1])[0]
        
        if best_strategy != self.current_strategy:
            logger.info(
                f"Agent {self.agent_id} changing strategy from "
                f"{self.current_strategy} to {best_strategy}"
            )
            self.current_strategy = best_strategy

    def handle_interaction_outcome(
        self,
        action: SocialAction,
        outcome: float,
        target_response: Optional[SocialAction] = None
    ) -> None:
        """Process and learn from interaction outcomes."""
        # Update history
        self.action_history.append(action)
        if len(self.action_history) > self.memory_size:
            self.action_history.pop(0)
        
        # Update trust and reputation
        self._update_trust_score(action.target_id, outcome)
        self._update_reputation_score(action.target_id, outcome)
        
        # Update network
        self._update_social_network(action, outcome)
        
        # Update strategy performance
        self.strategy_performance[self.current_strategy] = (
            self.strategy_performance[self.current_strategy] * (1 - self.learning_rate) +
            outcome * self.learning_rate
        )
        
        # Record outcome
        self.interaction_outcomes.append((action.target_id, outcome))
        if len(self.interaction_outcomes) > self.memory_size:
            self.interaction_outcomes.pop(0)

    def propose_coalition(
        self,
        potential_members: List[str],
        context: Dict[str, Any]
    ) -> Optional[Coalition]:
        """Propose a new coalition formation."""
        if not self._should_form_coalition(potential_members, context):
            return None
        
        # Calculate initial trust matrix
        trust_matrix = {}
        for m1 in potential_members + [self.agent_id]:
            for m2 in potential_members + [self.agent_id]:
                if m1 != m2:
                    trust_matrix[(m1, m2)] = self.trust_scores.get(m2, 0.5)
        
        # Calculate initial hierarchy based on reputation
        hierarchy = self._calculate_coalition_hierarchy(
            potential_members + [self.agent_id]
        )
        
        # Create coalition
        coalition = Coalition(
            id=f"coalition_{random.randint(0, 1000000)}",
            members=set(potential_members + [self.agent_id]),
            formation_time=context.get('timestamp', 0.0),
            trust_matrix=trust_matrix,
            shared_resources=0.0,
            stability_score=1.0,
            hierarchy=hierarchy
        )
        
        return coalition

    def evaluate_coalition_stability(self, coalition: Coalition) -> float:
        """Evaluate the stability of a coalition."""
        if not coalition.members:
            return 0.0
        
        # Calculate average trust
        trust_scores = list(coalition.trust_matrix.values())
        avg_trust = sum(trust_scores) / len(trust_scores)
        
        # Calculate resource distribution fairness
        resource_share = coalition.shared_resources / len(coalition.members)
        fairness_score = min(1.0, resource_share / 100.0)
        
        # Calculate hierarchy satisfaction
        hierarchy_score = self._calculate_hierarchy_satisfaction(coalition)
        
        # Combine factors
        stability = (
            0.4 * avg_trust +
            0.3 * fairness_score +
            0.3 * hierarchy_score
        )
        
        return stability

    def _calculate_action_utility(
        self,
        action: str,
        target_id: str,
        context: Dict[str, Any]
    ) -> float:
        """Calculate utility of an action considering multiple factors."""
        # Base utility
        base_utility = {
            "cooperate": 0.7,
            "compete": 0.5,
            "share_resource": 0.6,
            "request_help": 0.4
        }.get(action, 0.3)
        
        # Trust factor
        trust_factor = self.trust_scores.get(target_id, 0.5)
        
        # Reputation factor
        reputation_factor = self.reputation_scores[target_id]
        
        # Context factors
        resource_scarcity = context.get('resource_scarcity', 0.5)
        social_pressure = context.get('social_pressure', 0.5)
        
        # Calculate modified utility
        utility = base_utility * (
            0.3 * trust_factor +
            0.2 * reputation_factor +
            0.3 * (1 - resource_scarcity) +
            0.2 * social_pressure
        )
        
        return utility

    def _calculate_resource_allocation(
        self,
        action_type: str,
        utility: float,
        context: Dict[str, Any]
    ) -> float:
        """Calculate resource amount to allocate to an action."""
        # Base allocation
        base_allocation = min(
            self.resources * 0.2,  # Don't use more than 20% of resources
            utility * 50  # Scale with utility
        )
        
        # Adjust for context
        resource_scarcity = context.get('resource_scarcity', 0.5)
        social_pressure = context.get('social_pressure', 0.5)
        
        # Modify allocation
        modified_allocation = base_allocation * (
            0.7 +  # Base factor
            0.2 * (1 - resource_scarcity) +  # Resource availability
            0.1 * social_pressure  # Social influence
        )
        
        return min(modified_allocation, self.resources)

    def _apply_cooperative_bias(
        self,
        action_utilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply cooperative strategy bias to action utilities."""
        return {
            action: utility * (1.2 if action in ['cooperate', 'share_resource']
                             else 0.8)
            for action, utility in action_utilities.items()
        }

    def _apply_competitive_bias(
        self,
        action_utilities: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply competitive strategy bias to action utilities."""
        return {
            action: utility * (1.2 if action in ['compete', 'defend']
                             else 0.8)
            for action, utility in action_utilities.items()
        }

    def _evaluate_strategy_performance(
        self,
        strategy: StrategyType
    ) -> float:
        """Evaluate the performance of a strategy."""
        if not self.interaction_outcomes:
            return 0.5
        
        # Get recent outcomes during strategy use
        strategy_outcomes = [
            outcome for i, (_, outcome) in enumerate(self.interaction_outcomes)
            if i > len(self.interaction_outcomes) - 20
        ]
        
        if not strategy_outcomes:
            return self.strategy_performance.get(strategy, 0.5)
        
        return sum(strategy_outcomes) / len(strategy_outcomes)

    def _analyze_context(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze context for strategy selection."""
        return {
            'resource_pressure': context.get('resource_scarcity', 0.5),
            'social_pressure': context.get('social_pressure', 0.5),
            'competition_level': context.get('competition_level', 0.5),
            'cooperation_benefit': context.get('cooperation_benefit', 0.5)
        }

    def _get_context_modifier(
        self,
        strategy: StrategyType,
        context_factors: Dict[str, float]
    ) -> float:
        """Get context-based modifier for strategy scoring."""
        if strategy == StrategyType.COOPERATIVE:
            return (
                1.0 +
                0.3 * context_factors['cooperation_benefit'] -
                0.2 * context_factors['competition_level']
            )
        elif strategy == StrategyType.COMPETITIVE:
            return (
                1.0 +
                0.3 * context_factors['competition_level'] -
                0.2 * context_factors['cooperation_benefit']
            )
        elif strategy == StrategyType.ADAPTIVE:
            return 1.0 + 0.2 * context_factors['social_pressure']
        else:
            return 1.0

    def _update_trust_score(self, target_id: str, outcome: float) -> None:
        """Update trust score based on interaction outcome."""
        current_trust = self.trust_scores.get(target_id, 0.5)
        self.trust_scores[target_id] = (
            current_trust * (1 - self.learning_rate) +
            outcome * self.learning_rate
        )

    def _update_reputation_score(self, target_id: str, outcome: float) -> None:
        """Update reputation score based on interaction outcome."""
        current_reputation = self.reputation_scores[target_id]
        self.reputation_scores[target_id] = (
            current_reputation * (1 - self.learning_rate) +
            outcome * self.learning_rate
        )

    def _update_social_network(
        self,
        action: SocialAction,
        outcome: float
    ) -> None:
        """Update social network based on interaction."""
        if not self.social_network.has_edge(action.agent_id, action.target_id):
            self.social_network.add_edge(
                action.agent_id,
                action.target_id,
                weight=outcome
            )
        else:
            current_weight = self.social_network.edges[
                action.agent_id, action.target_id
            ]['weight']
            new_weight = (
                current_weight * (1 - self.learning_rate) +
                outcome * self.learning_rate
            )
            self.social_network.edges[
                action.agent_id, action.target_id
            ]['weight'] = new_weight

    def _should_form_coalition(
        self,
        potential_members: List[str],
        context: Dict[str, Any]
    ) -> bool:
        """Determine if coalition formation is beneficial."""
        # Check trust levels
        trust_levels = [
            self.trust_scores.get(member, 0.5)
            for member in potential_members
        ]
        avg_trust = sum(trust_levels) / len(trust_levels)
        
        # Check reputation
        reputation_levels = [
            self.reputation_scores[member]
            for member in potential_members
        ]
        avg_reputation = sum(reputation_levels) / len(reputation_levels)
        
        # Consider context
        resource_scarcity = context.get('resource_scarcity', 0.5)
        competition_level = context.get('competition_level', 0.5)
        
        # Calculate formation score
        formation_score = (
            0.3 * avg_trust +
            0.2 * avg_reputation +
            0.3 * (1 - resource_scarcity) +
            0.2 * competition_level
        )
        
        return formation_score > self.trust_threshold

    def _calculate_coalition_hierarchy(
        self,
        members: List[str]
    ) -> Dict[str, int]:
        """Calculate social hierarchy within coalition based on member influence."""
        influence_scores = {}
        
        for member in members:
            # Calculate base influence from reputation and trust
            reputation = self.reputation_scores[member]
            trust = sum(
                self.trust_scores.get(other, 0.5)
                for other in members if other != member
            ) / (len(members) - 1)
            
            # Consider network centrality if member is in network
            if member in self.social_network:
                centrality = nx.degree_centrality(self.social_network)[member]
                betweenness = nx.betweenness_centrality(self.social_network)[member]
            else:
                centrality = 0.0
                betweenness = 0.0
            
            # Combine factors into final influence score
            influence_scores[member] = (
                0.3 * reputation +
                0.3 * trust +
                0.2 * centrality +
                0.2 * betweenness
            )
        
        # Rank members by influence
        sorted_members = sorted(
            influence_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Assign hierarchy ranks (0 is highest)
        return {
            member: rank
            for rank, (member, _) in enumerate(sorted_members)
        }

    def _calculate_hierarchy_satisfaction(
        self,
        coalition: Coalition
    ) -> float:
        """Calculate member satisfaction with coalition hierarchy."""
        if not coalition.members:
            return 0.0
        
        satisfaction_scores = []
        
        for member in coalition.members:
            # Get member's current rank
            current_rank = coalition.hierarchy[member]
            
            # Calculate expected rank based on reputation and trust
            reputation = self.reputation_scores[member]
            avg_trust = sum(
                coalition.trust_matrix.get((member, other), 0.5)
                for other in coalition.members if other != member
            ) / (len(coalition.members) - 1)
            
            expected_rank = round(
                (1 - reputation) * (len(coalition.members) - 1) * 0.6 +
                (1 - avg_trust) * (len(coalition.members) - 1) * 0.4
            )
            
            # Calculate rank satisfaction (closer to expected = higher satisfaction)
            rank_diff = abs(current_rank - expected_rank)
            max_diff = len(coalition.members) - 1
            rank_satisfaction = 1 - (rank_diff / max_diff)
            
            satisfaction_scores.append(rank_satisfaction)
        
        # Return average satisfaction
        return sum(satisfaction_scores) / len(satisfaction_scores)

    def get_coalition_metrics(self) -> Dict[str, float]:
        """Get current coalition performance metrics."""
        if not self.current_coalition:
            return {}
        
        avg_trust = sum(
            self.current_coalition.trust_matrix.values()
        ) / len(self.current_coalition.trust_matrix)
        
        resource_efficiency = (
            self.current_coalition.shared_resources /
            (len(self.current_coalition.members) * 100)
        )
        
        hierarchy_satisfaction = self._calculate_hierarchy_satisfaction(
            self.current_coalition
        )
        
        return {
            'stability': self.current_coalition.stability_score,
            'average_trust': avg_trust,
            'resource_efficiency': resource_efficiency,
            'hierarchy_satisfaction': hierarchy_satisfaction,
            'member_count': len(self.current_coalition.members)
        }

    def analyze_social_position(self) -> Dict[str, float]:
        """Analyze agent's position in the social network."""
        if not self.social_network:
            return {}
        
        metrics = {}
        
        # Calculate network centrality metrics
        metrics['degree_centrality'] = nx.degree_centrality(
            self.social_network
        )[self.agent_id]
        
        metrics['betweenness_centrality'] = nx.betweenness_centrality(
            self.social_network
        )[self.agent_id]
        
        metrics['closeness_centrality'] = nx.closeness_centrality(
            self.social_network
        )[self.agent_id]
        
        # Calculate average relationship strength
        edges = self.social_network.edges(self.agent_id, data=True)
        if edges:
            metrics['avg_relationship_strength'] = sum(
                edge[2]['weight'] for edge in edges
            ) / len(edges)
        else:
            metrics['avg_relationship_strength'] = 0.0
        
        # Calculate social influence
        metrics['social_influence'] = (
            0.4 * metrics['degree_centrality'] +
            0.3 * metrics['betweenness_centrality'] +
            0.3 * metrics['avg_relationship_strength']
        )
        
        return metrics

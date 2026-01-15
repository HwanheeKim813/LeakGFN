"""
Improved Metrics Module for GFlowMoA
Provides evaluation metrics for molecular generation models.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from tqdm import tqdm
from scipy import stats
from rdkit.Chem import AllChem
from rdkit import DataStructs
from botorch.utils.multi_objective.hypervolume import Hypervolume

# ============================================================================
# Configuration and Constants
# ============================================================================

class MetricConstants:
    """Central repository for metric calculation constants."""
    # Molecular fingerprint settings
    MORGAN_RADIUS = 3
    MORGAN_BITS = 2048
    
    # Similarity thresholds
    NOVELTY_THRESHOLD = 0.4
    
    # Model constraints
    DEFAULT_MAX_BLOCKS = 15
    DEFAULT_MIN_BLOCKS = 8
    FALLBACK_MIN_BLOCKS = 8  # Used in correlation calculations
    
    # Numerical stability
    EPSILON = 1e-10
    LOG_EPSILON = -1000.0
    
    # Batch processing
    DEFAULT_BATCH_SIZE = 1000


@dataclass
class MetricConfig:
    """Configuration for metric calculations."""
    max_blocks: int = MetricConstants.DEFAULT_MAX_BLOCKS
    min_blocks: int = MetricConstants.DEFAULT_MIN_BLOCKS
    morgan_radius: int = MetricConstants.MORGAN_RADIUS
    morgan_bits: int = MetricConstants.MORGAN_BITS
    novelty_threshold: float = MetricConstants.NOVELTY_THRESHOLD
    batch_size: int = MetricConstants.DEFAULT_BATCH_SIZE
    verbose: bool = False
    
    @classmethod
    def from_args(cls, args: Any) -> 'MetricConfig':
        """Create config from argparse namespace."""
        return cls(
            max_blocks=getattr(args, 'max_blocks', cls.max_blocks),
            min_blocks=getattr(args, 'min_blocks', cls.min_blocks),
            verbose=getattr(args, 'verbose', False)
        )


# ============================================================================
# Utility Functions
# ============================================================================

class FingerprintCalculator:
    """Handles molecular fingerprint calculations."""
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_fingerprints(self, molecules: List[Any]) -> List:
        """
        Calculate Morgan fingerprints for molecules.
        
        Args:
            molecules: List of molecule objects with .mol attribute
            
        Returns:
            List of fingerprint objects
        """
        fps = []
        for mol in molecules:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol.mol if hasattr(mol, 'mol') else mol,
                    self.config.morgan_radius,
                    self.config.morgan_bits
                )
                fps.append(fp)
            except Exception as e:
                self.logger.warning(f"Failed to calculate fingerprint: {e}")
                continue
        return fps
    
    def calculate_pairwise_similarities(self, fps: List) -> np.ndarray:
        """
        Calculate pairwise Tanimoto similarities efficiently.
        
        Args:
            fps: List of fingerprints
            
        Returns:
            Array of similarity values
        """
        n = len(fps)
        if n <= 1:
            return np.array([])
        
        # Pre-allocate for efficiency
        similarities = []
        
        # Process in batches for memory efficiency
        batch_size = min(self.config.batch_size, n)
        
        for i in range(n):
            if i > 0:
                # Calculate similarities only with previous molecules
                batch_sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
                similarities.extend(batch_sims)
        
        return np.array(similarities)


# ============================================================================
# Core Metric Classes
# ============================================================================

class DiversityCalculator:
    """Calculates molecular diversity metrics."""
    
    def __init__(self, config: MetricConfig = None):
        self.config = config or MetricConfig()
        self.fp_calculator = FingerprintCalculator(self.config)
        self.logger = logging.getLogger(__name__)
    
    def compute(self, molecules: List[Any]) -> float:
        """
        Calculate diversity of molecules based on Tanimoto similarity.
        
        Args:
            molecules: List of molecule objects
            
        Returns:
            Diversity score between 0 and 1 (1 = most diverse)
        """
        if not molecules:
            return 0.0
        
        if len(molecules) == 1:
            return 1.0
        
        if self.config.verbose:
            self.logger.info(f"Computing diversity for {len(molecules)} molecules...")
        
        try:
            fps = self.fp_calculator.calculate_fingerprints(molecules)
            if len(fps) < 2:
                return 0.0
            
            similarities = self.fp_calculator.calculate_pairwise_similarities(fps)
            
            if len(similarities) == 0:
                return 1.0
            
            return float(1.0 - np.mean(similarities))
            
        except Exception as e:
            self.logger.error(f"Error computing diversity: {e}")
            return 0.0

class UniquenessCalculator:
    """Calculates molecular uniqueness metrics."""
    
    def __init__(self, config: MetricConfig = None):
        self.config = config or MetricConfig()
        self.logger = logging.getLogger(__name__)
    
    def compute(self, molecules: List[Any]) -> float:
        """
        Calculate uniqueness of molecules.
        
        Args:
            molecules: List of molecule objects
            
        Returns:
            Uniqueness score between 0 and 1 (1 = most unique)
        """
        smiles_set = set()
        for mol in molecules:
            smiles_set.add(mol.smiles)
        return len(smiles_set) / len(molecules)

class NoveltyCalculator:
    """Calculates molecular novelty metrics."""
    
    def __init__(self, config: MetricConfig = None):
        self.config = config or MetricConfig()
        self.fp_calculator = FingerprintCalculator(self.config)
        self.logger = logging.getLogger(__name__)
    
    def compute(self, molecules: List[Any], reference_molecules: List[Any]) -> float:
        """
        Calculate novelty of molecules compared to reference set.
        
        Args:
            molecules: List of molecule objects to evaluate
            reference_molecules: List of reference molecule objects
            
        Returns:
            Novelty score between 0 and 1 (1 = completely novel)
        """
        if not molecules or not reference_molecules:
            return 1.0
        
        if self.config.verbose:
            self.logger.info(f"Computing novelty for {len(molecules)} molecules...")
        
        try:
            mol_fps = self.fp_calculator.calculate_fingerprints(molecules)
            ref_fps = self.fp_calculator.calculate_fingerprints(reference_molecules)
            
            if not mol_fps or not ref_fps:
                return 1.0
            
            n_similar = 0
            for fp in mol_fps:
                similarities = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
                if max(similarities) >= self.config.novelty_threshold:
                    n_similar += 1
            
            novelty = 1.0 - (n_similar / len(mol_fps))
            return float(novelty)
            
        except Exception as e:
            self.logger.error(f"Error computing novelty: {e}")
            return 1.0


class SuccessCalculator:
    """Calculates success metrics for multi-objective optimization."""
    
    def __init__(self, config: MetricConfig = None):
        self.config = config or MetricConfig()
        self.logger = logging.getLogger(__name__)
    
    def compute(
        self,
        molecules: List[Any],
        scores: List[Dict[str, float]],
        objectives: List[str],
        thresholds: Dict[str, float]
    ) -> Tuple[float, List[Any]]:
        """
        Calculate success rate and identify successful molecules.
        
        Args:
            molecules: List of molecule objects
            scores: List of score dictionaries for each molecule
            objectives: List of objective names
            thresholds: Success thresholds for each objective
            
        Returns:
            Tuple of (success_rate, successful_molecules)
        """
        if not molecules or not scores:
            return 0.0, []
        
        if self.config.verbose:
            self.logger.info(f"Computing success rate for {len(molecules)} molecules...")
        
        successful_molecules = []
        success_counts = {obj: 0 for obj in objectives}
        
        for mol, score in zip(molecules, scores):
            all_successful = True
            
            for obj in objectives:
                if obj in score and score[obj] >= thresholds.get(obj, 0.0):
                    success_counts[obj] += 1
                else:
                    all_successful = False
            
            if all_successful:
                successful_molecules.append(mol)
        
        success_rate = len(successful_molecules) / len(molecules)
        
        if self.config.verbose:
            for obj, count in success_counts.items():
                self.logger.info(f"  {obj}: {count}/{len(molecules)} successful")
        
        return float(success_rate), successful_molecules


# ============================================================================
# Evaluation Functions
# ============================================================================

class SingleObjectiveEvaluator:
    """Evaluator for single-objective molecular generation."""
    
    def __init__(self, config: MetricConfig = None):
        self.config = config or MetricConfig()
        self.diversity_calc = DiversityCalculator(self.config)
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        args: Any,
        generator: nn.Module,
        rollout_worker: Any,
        k: int,
        return_samples: bool = False
    ) -> Union[Tuple[float, float, float, float, List[float]],
               Tuple[float, float, float, float, List[float], List[Any]]]:
        """
        Evaluate generator performance for single objective.
        
        Args:
            args: Configuration arguments
            generator: Neural network generator
            rollout_worker: Worker for molecular rollouts
            k: Number of top molecules to consider
            return_samples: Whether to return sampled molecules
            
        Returns:
            Evaluation metrics tuple with optional samples
        """
        start_time = time.time()
        self.logger.info(f"Evaluating single-objective performance (k={k})...")
        
        # Sample molecules
        sampled_molecules = []
        rewards = []
        scores = []
        
        try:
            # for i in tqdm(range(args.num_samples), desc="Sampling molecules"):
            for i in range(args.num_samples):
                rollout_worker.rollout(generator, use_rand_policy=False)
                raw_r, mol, _, _ = rollout_worker.sampled_mols[-1]
                
                sampled_molecules.append(mol)
                rewards.append(float(raw_r[0]))
                scores.append(raw_r[1])
            
            # Select top-k molecules
            idx_top = np.argsort(rewards)[::-1][:k]
            top_molecules = [sampled_molecules[i] for i in idx_top]
            
            # Calculate metrics
            all_diversity = self.diversity_calc.compute(sampled_molecules)
            top_diversity = self.diversity_calc.compute(top_molecules)
            top_rewards = float(np.mean([rewards[i] for i in idx_top]))
            mean_reward = float(np.mean(rewards))
            mean_score = np.mean(scores, axis=0) if scores else 0.0
            
            # Log results
            elapsed_time = time.time() - start_time
            self._log_results(
                k, all_diversity, top_diversity, 
                top_rewards, mean_reward, mean_score, elapsed_time
            )
            
            if return_samples:
                return top_rewards, top_diversity, mean_reward, mean_score, rewards, sampled_molecules
            else:
                return top_rewards, top_diversity, mean_reward, mean_score, rewards
                
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            raise
    
    def _log_results(
        self, k: int, all_diversity: float, top_diversity: float,
        top_rewards: float, mean_reward: float, mean_score: Any, elapsed_time: float
    ):
        """Log evaluation results."""
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  All molecules - Mean reward: {mean_reward:.4f}")
        self.logger.info(f"  All molecules - Mean score: {mean_score}")
        self.logger.info(f"  All molecules - Diversity: {all_diversity:.4f}")
        self.logger.info(f"  Top-{k} - Rewards: {top_rewards:.4f}")
        self.logger.info(f"  Top-{k} - Diversity: {top_diversity:.4f}")
        self.logger.info(f"  Time elapsed: {elapsed_time:.2f}s")


class MultiObjectiveEvaluator:
    """Evaluator for multi-objective molecular generation."""
    
    def __init__(self, config: MetricConfig = None):
        self.config = config or MetricConfig()
        self.diversity_calc = DiversityCalculator(self.config)
        self.uniqueness_calc = UniquenessCalculator(self.config)
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        args: Any,
        generator: nn.Module,
        rollout_worker: Any,
        k: int,
        return_samples: bool = False
    ) -> Tuple[Tuple, Tuple, List[Dict], Optional[List[Any]]]:
        """
        Evaluate generator performance for multiple objectives.
        
        Args:
            args: Configuration arguments with objectives list
            generator: Neural network generator
            rollout_worker: Worker for molecular rollouts
            k: Number of top molecules to consider
            return_samples: Whether to return sampled molecules
            
        Returns:
            Tuple of (top_metrics, all_metrics, score_dicts, [samples])
        """
        start_time = time.time()
        self.logger.info(f"Evaluating multi-objective performance (k={k})...")
        
        # Initialize hypervolume calculator
        ref_point = torch.zeros(len(args.objectives))
        hypervolume_calc = Hypervolume(ref_point=ref_point)
        
        # Sample molecules
        sampled_molecules = []
        rewards = []
        scores = []
        score_dicts = []
        
        try:
            # for i in tqdm(range(args.num_samples), desc="Sampling molecules"):
            for i in range(args.num_samples):
                rollout_worker.rollout(generator, use_rand_policy=False)
                raw_reward_dict, mol, _, _ = rollout_worker.sampled_mols[-1]
                
                # Extract scores for each objective
                obj_scores = {
                    obj: raw_reward_dict[obj][2] 
                    for obj in args.objectives
                }
                score_dicts.append(obj_scores)
                
                # Calculate combined reward
                reward, _, _ = raw_reward_dict.get('combined', (0, 0, 0))
                log_reward = torch.log(reward + args.log_reg_c).item()
                
                rewards.append(log_reward)
                scores.append([obj_scores[obj].item() for obj in args.objectives])
                sampled_molecules.append(mol)
            
            # Calculate metrics for all molecules
            all_metrics = self._calculate_metrics(
                sampled_molecules, rewards, scores, 
                hypervolume_calc, "All molecules"
            )
            
            # Calculate metrics for top-k molecules
            idx_top = np.argsort(rewards)[::-1][:k]
            top_molecules = [sampled_molecules[i] for i in idx_top]
            top_rewards = [rewards[i] for i in idx_top]
            top_scores = [scores[i] for i in idx_top]
            
            top_metrics = self._calculate_metrics(
                top_molecules, top_rewards, top_scores,
                hypervolume_calc, f"Top-{k}"
            )
            
            # Log results
            elapsed_time = time.time() - start_time
            self._log_results(args.objectives, all_metrics, top_metrics, k, elapsed_time)
            
            if return_samples:
                return top_metrics, all_metrics, score_dicts, sampled_molecules, top_molecules
            else:
                return top_metrics, all_metrics, score_dicts
                
        except Exception as e:
            self.logger.error(f"Multi-objective evaluation failed: {e}")
            raise
    
    def _calculate_metrics(
        self,
        molecules: List[Any],
        rewards: List[float],
        scores: List[List[float]],
        hypervolume_calc: Hypervolume,
        label: str
    ) -> Tuple[float, np.ndarray, float, float]:
        """Calculate evaluation metrics."""
        diversity = self.diversity_calc.compute(molecules)
        uniqueness = self.uniqueness_calc.compute(molecules)
        mean_reward = float(np.mean(rewards))
        mean_scores = np.mean(scores, axis=0)
        
        # Calculate hypervolume
        try:
            hypervolume = hypervolume_calc.compute(torch.tensor(scores))
        except Exception as e:
            self.logger.warning(f"Hypervolume calculation failed: {e}")
            hypervolume = 0.0
        harmonic_mean = compute_harmonic_mean([diversity, uniqueness, hypervolume])
        return mean_reward, mean_scores, diversity, uniqueness, float(hypervolume), harmonic_mean
    
    def _log_results(
        self, 
        objectives: List[str],
        all_metrics: Tuple,
        top_metrics: Tuple,
        k: int,
        elapsed_time: float
    ):
        """Log evaluation results."""
        all_reward, all_scores, all_diversity, all_uniqueness, all_volume, all_hm = all_metrics
        top_reward, top_scores, top_diversity, top_uniqueness, top_volume, top_hm = top_metrics
        
        self.logger.info(f"Multi-Objective Evaluation Results:")
        self.logger.info(f"Objectives: {objectives}")
        self.logger.info(f"--------------------------------")
        self.logger.info(f"All molecules:")
        self.logger.info(f"Mean reward: {all_reward:.4f}")
        self.logger.info(f"Mean scores: {all_scores}")
        self.logger.info(f"Diversity: {all_diversity:.4f}")
        self.logger.info(f"Uniqueness: {all_uniqueness:.4f}")
        self.logger.info(f"Hypervolume: {all_volume:.4f}")
        self.logger.info(f"Harmonic mean: {all_hm:.4f}")
        self.logger.info(f"--------------------------------")
        self.logger.info(f"Top-{k} molecules:")
        self.logger.info(f"Mean reward: {top_reward:.4f}")
        self.logger.info(f"Mean scores: {top_scores}")
        self.logger.info(f"Diversity: {top_diversity:.4f}")
        self.logger.info(f"Uniqueness: {top_uniqueness:.4f}")
        self.logger.info(f"Hypervolume: {top_volume:.4f}")
        self.logger.info(f"Harmonic mean: {top_hm:.4f}")
        self.logger.info(f"--------------------------------")
        self.logger.info(f"Time elapsed: {elapsed_time:.2f}s")


def compute_harmonic_mean(scores):
    return 1/(np.mean(1/np.array(scores))).item()

# ============================================================================
# Correlation Analysis
# ============================================================================

class PathGraphBuilder:
    """Builds molecular path graphs for correlation analysis."""
    
    def __init__(self, mdp: Any):
        self.mdp = mdp
        self.logger = logging.getLogger(__name__)
    
    def build_path_graph(self, molecule: Any) -> nx.DiGraph:
        """
        Build a directed graph representing the construction path of a molecule.
        
        Args:
            molecule: Molecule object
            
        Returns:
            NetworkX directed graph with molecule construction path
            
        Raises:
            ValueError: If graph construction fails
        """
        try:
            graph = nx.DiGraph()
            graph.add_node(0)
            
            ancestors = [molecule]
            ancestor_graphs = []
            
            # Get parent molecules
            parents = self.mdp.parents(molecule)
            mol_stack = [p[0] for p in parents]
            parent_stack = [[0, action] for p, action in parents]
            
            while mol_stack:
                current_mol = mol_stack.pop()
                parent_node, parent_action = parent_stack.pop()
                
                # Check for isomorphic graphs
                current_graph = self.mdp.get_nx_graph(current_mol)
                match_found = False
                
                for idx, ancestor_graph in enumerate(ancestor_graphs):
                    if self.mdp.graphs_are_isomorphic(current_graph, ancestor_graph):
                        graph.add_edge(parent_node, idx + 1, action=parent_action)
                        match_found = True
                        break
                
                if not match_found:
                    new_node_id = len(ancestors)
                    graph.add_edge(parent_node, new_node_id, action=parent_action)
                    ancestors.append(current_mol)
                    ancestor_graphs.append(current_graph)
                    
                    if len(current_mol.blocks) > 0:
                        new_parents = self.mdp.parents(current_mol)
                        mol_stack.extend([p[0] for p in new_parents])
                        parent_stack.extend(
                            [(new_node_id, action) for p, action in new_parents]
                        )
            
            # Verify and fix actions
            self._verify_graph_actions(graph, ancestors)
            
            # Store molecules in graph nodes
            for node_id in graph.nodes:
                graph.nodes[node_id]['mol'] = ancestors[node_id]
            
            return graph
            
        except Exception as e:
            self.logger.error(f"Failed to build path graph: {e}")
            raise ValueError(f"Path graph construction failed: {e}")
    
    def _verify_graph_actions(self, graph: nx.DiGraph, ancestors: List[Any]):
        """Verify and fix graph edge actions."""
        for u, v in graph.edges:
            action = graph.edges[(u, v)]['action']
            constructed = self.mdp.add_block_to(ancestors[v], *action)
            
            target_graph = self.mdp.get_nx_graph(ancestors[u], true_block=True)
            constructed_graph = self.mdp.get_nx_graph(constructed, true_block=True)
            
            if not self.mdp.graphs_are_isomorphic(constructed_graph, target_graph):
                # Try to fix the action
                block, _ = action
                for stem_idx in range(len(ancestors[v].stems)):
                    test_mol = self.mdp.add_block_to(ancestors[v], block, stem_idx)
                    test_graph = self.mdp.get_nx_graph(test_mol, true_block=True)
                    
                    if self.mdp.graphs_are_isomorphic(test_graph, target_graph):
                        graph.edges[(u, v)]['action'] = (block, stem_idx)
                        break
                else:
                    raise ValueError(f"Could not fix action for edge ({u}, {v})")


class CorrelationCalculator:
    """Base class for correlation calculations."""
    
    def __init__(self, mdp: Any, device: str, config: MetricConfig = None):
        self.mdp = mdp
        self.device = device
        self.config = config or MetricConfig()
        self.logsoftmax = nn.LogSoftmax(0)
        self.path_builder = PathGraphBuilder(mdp)
        self.logger = logging.getLogger(__name__)
    
    def _tensor(self, x: Any) -> torch.Tensor:
        """Convert to tensor on correct device."""
        return torch.tensor(x, device=self.device, dtype=torch.float)
    
    def _calculate_action_logprobs(
        self,
        graph: nx.DiGraph,
        model: nn.Module,
        mol: Any,
        objective: str = 'combined'
    ) -> nx.DiGraph:
        """
        Calculate log probabilities for actions in the path graph.
        
        Args:
            graph: Path graph
            model: Neural network model
            mol: Target molecule
            objective: Objective to evaluate
            
        Returns:
            Graph with logprob annotations
        """
        # Get representations for all molecules in graph
        mol_reprs = [graph.nodes[i]['mol'] for i in graph.nodes]
        states = self.mdp.mols2batch([self.mdp.mol2repr(m) for m in mol_reprs])
        
        with torch.no_grad():
            outputs = model(states)
            stem_out, mol_out = outputs.get(objective, outputs)
        
        # Calculate action probabilities for each node
        per_mol_outputs = []
        for node_idx in range(len(graph.nodes)):
            mol_obj = graph.nodes[node_idx]['mol']
            stem_slice = states._slice_dict['stems'][node_idx:node_idx+2]
            
            # Check if stop action is allowed
            stop_allowed = len(mol_obj.blocks) >= self.config.min_blocks
            
            # Calculate log probabilities
            logits = torch.cat([
                stem_out[stem_slice[0]:stem_slice[1]].reshape(-1),
                mol_out[node_idx, :1] if stop_allowed else self._tensor([MetricConstants.LOG_EPSILON])
            ])
            
            probs = self.logsoftmax(logits)
            stem_probs = probs[:-1].reshape(-1, stem_out.shape[1])
            stop_prob = probs[-1]
            
            per_mol_outputs.append((stem_probs, stop_prob))
        
        # Handle terminal states
        if len(mol.blocks) < self.config.max_blocks:
            with torch.no_grad():
                final_output = model(self.mdp.mols2batch([self.mdp.mol2repr(mol)]))
                final_stem, final_mol = final_output.get(objective, final_output)
                
                final_logits = torch.cat([
                    final_stem.reshape(-1),
                    final_mol[0, :1]
                ])
                final_probs = self.logsoftmax(final_logits)
                stop_prob = final_probs[-1]
        else:
            stop_prob = self._tensor(0.0)
        
        # Assign log probabilities to edges
        for u, v in graph.edges:
            action = graph.edges[u, v]['action']
            if action[0] == -1:  # Stop action
                graph.edges[u, v]['logprob'] = per_mol_outputs[v][1]
            else:
                graph.edges[u, v]['logprob'] = per_mol_outputs[v][0][action[1], action[0]]
        
        return graph, stop_prob
    
    def _propagate_logprobs(self, graph: nx.DiGraph, stop_prob: torch.Tensor) -> float:
        """
        Propagate log probabilities through the graph.
        
        Args:
            graph: Path graph with edge logprobs
            stop_prob: Terminal stop probability
            
        Returns:
            Total log probability
        """
        # Propagate in reverse topological order
        for node in reversed(list(nx.topological_sort(graph))):
            for parent in graph.predecessors(node):
                edge_logprob = graph.edges[parent, node]['logprob']
                node_logprob = graph.nodes[node].get('logprob', 0)
                
                if parent == 0 and len(graph.nodes[node]['mol'].blocks) < self.config.max_blocks:
                    # Include stop probability for root node
                    total_logprob = edge_logprob + node_logprob + stop_prob
                else:
                    total_logprob = edge_logprob + node_logprob
                
                # Update parent's logprob using logaddexp for numerical stability
                current_parent_logprob = graph.nodes[parent].get('logprob', self._tensor(MetricConstants.LOG_EPSILON))
                graph.nodes[parent]['logprob'] = torch.logaddexp(
                    current_parent_logprob,
                    total_logprob
                )
        
        # Return root node's total logprob
        return graph.nodes[0]['logprob'].item()


class SingleObjectiveCorrelation(CorrelationCalculator):
    """Correlation calculator for single objective optimization."""
    
    def compute(
        self,
        model: nn.Module,
        rollout_worker: Any,
        test_molecules: List[Any],
        objective: str = 'combined'
    ) -> float:
        """
        Compute Spearman correlation for single objective.
        
        Args:
            model: Neural network model
            rollout_worker: Rollout worker with reward function
            test_molecules: List of test molecules
            objective: Objective to evaluate
            
        Returns:
            Spearman correlation coefficient
        """
        start_time = time.time()
        self.logger.info(f"Computing correlation for {len(test_molecules)} molecules...")
        
        rewards = []
        log_probs = []
        failed_count = 0
        
        # for mol in tqdm(test_molecules, desc="Processing molecules"):
        for mol in test_molecules:
            try:
                # Build path graph
                graph = self.path_builder.build_path_graph(mol)
                
                # Calculate reward
                reward = rollout_worker._get_reward(mol, model)[0]
                if isinstance(reward, dict):
                    reward = reward.get(objective, 0.0)
                rewards.append(np.log(reward + MetricConstants.EPSILON).item())
                
                # Calculate log probabilities
                graph, stop_prob = self._calculate_action_logprobs(
                    graph, model, mol, objective
                )
                
                # Propagate and get total logprob
                total_logprob = self._propagate_logprobs(graph, stop_prob)
                log_probs.append(total_logprob)
                
            except Exception as e:
                self.logger.debug(f"Failed to process molecule: {e}")
                failed_count += 1
                continue
        
        if len(rewards) < 2:
            self.logger.warning("Not enough valid molecules for correlation")
            return 0.0
        
        # Calculate correlation
        correlation = stats.spearmanr(rewards, log_probs).correlation
        
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Correlation: {correlation:.4f} "
            f"(processed {len(rewards)}/{len(test_molecules)} molecules, "
            f"failed: {failed_count}, time: {elapsed_time:.2f}s)"
        )
        
        return float(correlation)


class MultiObjectiveCorrelation(CorrelationCalculator):
    """Correlation calculator for multi-objective optimization."""
    
    def compute(
        self,
        model: nn.Module,
        rollout_worker: Any,
        test_molecules: List[Any],
        objectives: List[str]
    ) -> Dict[str, float]:
        """
        Compute Spearman correlations for multiple objectives.
        
        Args:
            model: Neural network model
            rollout_worker: Rollout worker with reward function
            test_molecules: List of test molecules
            objectives: List of objective names
            
        Returns:
            Dictionary of correlations per objective
        """
        start_time = time.time()
        self.logger.info(
            f"Computing multi-objective correlations for {len(test_molecules)} molecules..."
        )
        
        # Initialize storage for each objective
        rewards_dict = {obj: [] for obj in objectives}
        rewards_dict['combined'] = []
        logprobs_dict = {obj: [] for obj in objectives}
        logprobs_dict['combined'] = []
        
        failed_count = 0
        
        # for mol in tqdm(test_molecules, desc="Processing molecules"):
        for mol in test_molecules:
            try:
                # Build path graph once
                graph = self.path_builder.build_path_graph(mol)
                
                # Calculate rewards
                reward_data = rollout_worker._get_reward(mol, model)[0]
                
                # Process each objective
                for obj in objectives + ['combined']:
                    if obj in reward_data:
                        reward = reward_data[obj]
                        rewards_dict[obj].append(np.log(reward + MetricConstants.EPSILON).item())
                        
                        # Calculate log probabilities for this objective
                        obj_graph = graph.copy()
                        obj_graph, stop_prob = self._calculate_action_logprobs(
                            obj_graph, model, mol, obj
                        )
                        
                        total_logprob = self._propagate_logprobs(obj_graph, stop_prob)
                        logprobs_dict[obj].append(total_logprob)
                
            except Exception as e:
                self.logger.debug(f"Failed to process molecule: {e}")
                failed_count += 1
                continue
        
        # Calculate correlations for each objective
        correlations = {}
        for obj in rewards_dict:
            if len(rewards_dict[obj]) >= 2:
                corr = stats.spearmanr(rewards_dict[obj], logprobs_dict[obj]).correlation
                correlations[obj] = float(corr)
            else:
                correlations[obj] = 0.0
        
        elapsed_time = time.time() - start_time
        
        # Log results
        mean_corr = np.mean(list(correlations.values()))
        self.logger.info(f"Multi-objective correlations: {correlations}")
        self.logger.info(
            f"Mean correlation: {mean_corr:.4f} "
            f"(failed: {failed_count}, time: {elapsed_time:.2f}s)"
        )
        
        return correlations


# ============================================================================
# Public API Functions (for backward compatibility)
# ============================================================================

def compute_diversity(molecules: List[Any], verbose: bool = False) -> float:
    """
    Calculate diversity of molecules.
    
    Args:
        molecules: List of molecule objects
        verbose: Whether to print progress
        
    Returns:
        Diversity score between 0 and 1
    """
    config = MetricConfig(verbose=verbose)
    calculator = DiversityCalculator(config)
    return calculator.compute(molecules)


def compute_novelty(
    molecules: List[Any],
    reference_molecules: List[Any],
    verbose: bool = False
) -> float:
    """
    Calculate novelty of molecules compared to reference set.
    
    Args:
        molecules: List of molecule objects
        reference_molecules: List of reference molecules
        verbose: Whether to print progress
        
    Returns:
        Novelty score between 0 and 1
    """
    config = MetricConfig(verbose=verbose)
    calculator = NoveltyCalculator(config)
    return calculator.compute(molecules, reference_molecules)


def compute_success(
    molecules: List[Any],
    scores: List[Dict[str, float]],
    objectives: List[str],
    thresholds: Dict[str, float],
    verbose: bool = False
) -> Tuple[float, List[Any]]:
    """
    Calculate success rate for multi-objective optimization.
    
    Args:
        molecules: List of molecule objects
        scores: List of score dictionaries
        objectives: List of objective names
        thresholds: Success thresholds per objective
        verbose: Whether to print progress
        
    Returns:
        Tuple of (success_rate, successful_molecules)
    """
    config = MetricConfig(verbose=verbose)
    calculator = SuccessCalculator(config)
    return calculator.compute(molecules, scores, objectives, thresholds)


def evaluate(
    args: Any,
    generator: nn.Module,
    rollout_worker: Any,
    k: int,
    return_samples: bool = False
) -> Union[Tuple[float, float, float, float, List[float]],
           Tuple[float, float, float, float, List[float], List[Any]]]:
    """
    Evaluate single-objective generator performance.
    
    Args:
        args: Configuration arguments
        generator: Neural network generator
        rollout_worker: Rollout worker
        k: Number of top molecules
        return_samples: Whether to return samples
        
    Returns:
        Evaluation metrics tuple
    """
    config = MetricConfig.from_args(args)
    evaluator = SingleObjectiveEvaluator(config)
    return evaluator.evaluate(args, generator, rollout_worker, k, return_samples)


def evaluate_MOO(
    args: Any,
    generator: nn.Module,
    rollout_worker: Any,
    k: int,
    return_samples: bool = False
) -> Tuple[Tuple, Tuple, List[Dict], Optional[List[Any]]]:
    """
    Evaluate multi-objective generator performance.
    
    Args:
        args: Configuration arguments with objectives
        generator: Neural network generator
        rollout_worker: Rollout worker
        k: Number of top molecules
        return_samples: Whether to return samples
        
    Returns:
        Tuple of metrics and samples
    """
    config = MetricConfig.from_args(args)
    evaluator = MultiObjectiveEvaluator(config)
    return evaluator.evaluate(args, generator, rollout_worker, k, return_samples)


def compute_correlation(
    args: Any,
    model: nn.Module,
    rollout_worker: Any,
    test_molecules: List[Any],
    key: str = 'combined'
) -> float:
    """
    Compute single-objective correlation.
    
    Args:
        args: Configuration arguments
        model: Neural network model
        rollout_worker: Rollout worker
        test_molecules: Test molecules
        key: Objective key
        
    Returns:
        Spearman correlation coefficient
    """
    config = MetricConfig.from_args(args)
    calculator = SingleObjectiveCorrelation(
        rollout_worker.mdp,
        args.device,
        config
    )
    return calculator.compute(model, rollout_worker, test_molecules, key)


def compute_correlation_moa(
    args: Any,
    model: nn.Module,
    rollout_worker: Any,
    test_molecules: List[Any]
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    Compute multi-objective correlations.
    
    Args:
        args: Configuration arguments with objectives
        model: Neural network model
        rollout_worker: Rollout worker
        test_molecules: Test molecules
        
    Returns:
        Tuple of (correlations, rewards_dict)
    """
    config = MetricConfig.from_args(args)
    calculator = MultiObjectiveCorrelation(
        rollout_worker.mdp,
        args.device,
        config
    )
    
    correlations = calculator.compute(
        model,
        rollout_worker,
        test_molecules,
        args.objectives
    )
    
    # For backward compatibility, return rewards dict
    # (Note: This is a simplified version, actual implementation may need adjustment)
    rewards_dict = {obj: [] for obj in args.objectives}
    rewards_dict['combined'] = []
    
    return correlations, rewards_dict


def get_mol_path_graph(molecule: Any, mdp: Any) -> nx.DiGraph:
    """
    Build molecular path graph.
    
    Args:
        molecule: Molecule object
        mdp: MDP object
        
    Returns:
        NetworkX directed graph
    """
    builder = PathGraphBuilder(mdp)
    return builder.build_path_graph(molecule)


# ============================================================================
# Setup Logging
# ============================================================================

def setup_logging(level: str = "INFO"):
    """Setup module logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Initialize logging when module is imported
setup_logging()
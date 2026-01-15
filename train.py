"""
GFlowMoA Training Script - Improved Version
Multi-objective molecular optimization using GFlowNet
Author: Improved version with bug fixes and best practices
"""

import os
import time
import threading
import pickle
import gzip
import warnings
import datetime
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

import torch
import numpy as np
import wandb
from rdkit.Chem import AllChem
from rdkit import DataStructs

from gflownet.mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from gflownet.generator import LeakFMGFlowNet, TBGFlowNet, FMGFlowNet, MOReinforce, SubTBGFlowNet
from gflownet.oracle.oracle import Oracle
from gflownet.utils.metrics import evaluate, compute_correlation
from gflownet.utils.utils import set_random_seed
from gflownet.utils.logging import get_logger
from gflownet.utils.arguments import argparser
import pandas as pd
warnings.filterwarnings('ignore')

# ============================================================================
# Constants
# ============================================================================
class Constants:
    """Central repository for all constants used in the training process"""
    TEST_SEED = 142857
    DEFAULT_MAX_SAMPLED_MOLS = 10000
    DEFAULT_MAX_ONLINE_MOLS = 1000
    DEFAULT_MAX_HINDSIGHT_MOLS = 1000
    DEFAULT_SAMPLE_ITERATIONS = 100
    THREAD_POOL_SIZE = 8
    GPU_MEMORY_CLEANUP_INTERVAL = 1000
    LOG_INTERVAL = 100
    CHECKPOINT_PREFIX = "checkpoint"
    
    # Thresholds for different objectives
    OBJECTIVE_THRESHOLDS = {
        'sa': 0.7,
        'qed': 0.6, 
        'jnk3': 0.5,
        'gsk3b': 0.5,
        'drd2': 0.5,
        'seh': 8
    }

# ============================================================================
# Configuration
# ============================================================================
@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    seed: int
    device: str
    num_iterations: int
    trajectories_mbsize: int
    min_blocks: int
    max_blocks: int
    reward_exp: float
    reward_min: float
    reward_bin: float
    reward_norm: float
    reward_exp_ramping: float
    random_action_prob: float
    criterion: str
    sample_iterations: int
    save: bool
    debug: bool
    log_dir: str
    objectives: List[str]
    
    @classmethod
    def from_args(cls, args):
        """Create config from argparse namespace"""
        return cls(
            seed=args.seed,
            device=args.device,
            num_iterations=args.num_iterations,
            trajectories_mbsize=args.trajectories_mbsize,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            reward_exp=args.reward_exp,
            reward_min=args.reward_min,
            reward_bin=args.reward_bin,
            reward_norm=args.reward_norm,
            reward_exp_ramping=args.reward_exp_ramping,
            random_action_prob=args.random_action_prob,
            criterion=args.criterion,
            sample_iterations=args.sample_iterations,
            save=args.save,
            debug=args.debug,
            log_dir=args.log_dir,
            objectives=args.objectives
        )

# ============================================================================
# Exception Classes
# ============================================================================
class MoleculeGenerationError(Exception):
    """Exception raised when molecule generation fails"""
    pass

class RewardCalculationError(Exception):
    """Exception raised when reward calculation fails"""
    pass

# ============================================================================
# RolloutWorker Class - Improved Version
# ============================================================================
class RolloutWorker:
    """
    Worker class for generating molecular trajectories and computing rewards.
    Thread-safe implementation with proper memory management.
    """
    
    def __init__(self, args, bpath: str, proxy: Oracle, device: str):
        """
        Initialize the rollout worker.
        
        Args:
            args: Configuration arguments
            bpath: Path to molecular building blocks
            proxy: Oracle for scoring molecules
            device: Device to run computations on
        """
        self.args = args
        self.config = TrainingConfig.from_args(args) if not isinstance(args, TrainingConfig) else args
        self.logger = logging.getLogger(__name__)
        
        # Random number generators
        self.test_split_rng = np.random.RandomState(Constants.TEST_SEED)
        self.train_rng = np.random.RandomState(int(time.time()))
        
        # Initialize MDP
        self._init_mdp(bpath, device, args)
        
        # Initialize proxy
        self.proxy = proxy
        self._device = device
        
        # Thread safety
        self._mol_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        
        # Molecule storage with memory management
        self.seen_molecules = set()
        self.sampled_mols = []
        self.online_mols = []
        self.hindsight_mols = []
        
        # Storage limits
        self.max_sampled_mols = Constants.DEFAULT_MAX_SAMPLED_MOLS
        self.max_online_mols = Constants.DEFAULT_MAX_ONLINE_MOLS
        self.max_hindsight_mols = Constants.DEFAULT_MAX_HINDSIGHT_MOLS
        
        # Training parameters
        self._init_training_params(args)
        
        # Threading control
        self.stop_event = threading.Event()
        
    def _init_mdp(self, bpath: str, device: str, args):
        """Initialize the molecular MDP"""
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, args.repr_type, include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        
        # Set float precision
        if args.floatX == 'float64':
            self.mdp.floatX = self.floatX = torch.double
        else:
            self.mdp.floatX = self.floatX = torch.float
            
    def _init_training_params(self, args):
        """Initialize training-specific parameters"""
        self.min_blocks = args.min_blocks
        self.max_blocks = args.max_blocks
        self.mdp._cue_max_blocks = self.max_blocks
        
        self.reward_exp = args.reward_exp
        self.reward_min = args.reward_min
        self.reward_bin = args.reward_bin
        self.reward_norm = args.reward_norm
        self.reward_exp_ramping = args.reward_exp_ramping
        self.random_action_prob = args.random_action_prob
        # Set ignore_parents based on criterion
        if args.criterion in ['TB', 'Reinforce', 'SubTB']:
            self.ignore_parents = True
        elif args.criterion in ['FM', 'LeakGFN']:
            self.ignore_parents = False
        else:
            raise ValueError(f"Unknown criterion: {args.criterion}")
    
    def _add_sampled_mol(self, mol_data: Tuple):
        """
        Thread-safe addition of sampled molecules with memory management.
        
        Args:
            mol_data: Tuple containing molecule information
        """
        with self._mol_lock:
            if len(self.sampled_mols) >= self.max_sampled_mols:
                # Remove oldest molecules (FIFO)
                self.sampled_mols = self.sampled_mols[-self.max_sampled_mols//2:]
            self.sampled_mols.append(mol_data)
    
    def _add_mol_to_replay(self, mol: BlockMoleculeDataExtended):
        """
        Add molecule to replay buffer for hindsight experience replay.
        
        Args:
            mol: Molecule to add to replay buffer
        """
        with self._mol_lock:
            if len(self.hindsight_mols) >= self.max_hindsight_mols:
                # Random replacement strategy
                idx = self.train_rng.randint(0, len(self.hindsight_mols))
                self.hindsight_mols[idx] = mol
            else:
                self.hindsight_mols.append(mol)
    
    def _get_reward(self, m: BlockMoleculeDataExtended) -> Tuple[float, Tuple]:
        """
        Calculate reward for a molecule with proper error handling.
        
        Args:
            m: Molecule to evaluate
            
        Returns:
            Tuple of (processed_reward, (raw_reward, score))
            
        Raises:
            RewardCalculationError: If reward calculation fails
        """
        try:
            rdmol = m.mol
            if rdmol is None:
                self.logger.warning(f"Invalid molecule generated: {m}")
                return self.l2r(self.reward_min), (0, 0)
            
            # Get scores from oracle
            score = self.proxy.get_score(m)
            if not isinstance(score, dict):
                raise RewardCalculationError(f"Invalid score format: {score}")
                
            score_tensor = torch.tensor(list(score.values())).to(self._device)
            
            # Calculate raw reward
            raw_reward = score_tensor.sum()
            
            # Apply reward binarization with fixed logic
            # raw_reward = self._apply_reward_binarization(raw_reward)
            
            # Transform to final reward
            reward = self.l2r(raw_reward.clip(self.reward_min))
            
            return reward, (raw_reward, score_tensor)
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return self.l2r(self.reward_min), (0, 0)
    
    def _apply_reward_binarization(self, raw_reward: torch.Tensor) -> torch.Tensor:
        """
        Apply reward binarization with corrected logic.
        
        Args:
            raw_reward: Raw reward tensor
            
        Returns:
            Binarized reward tensor
        """
        # Set rewards >= reward_bin to 1 (maximum reward)
        # raw_reward[raw_reward >= self.reward_bin] += (self.reward_norm*10)-self.reward_bin
        raw_reward[raw_reward >= self.reward_bin] = 1
        # Apply normalization for intermediate rewards (fixed logic)
        if self.reward_norm != self.reward_bin:
            mask = (raw_reward < self.reward_bin) & (raw_reward > self.reward_norm)
            raw_reward[mask] = self.reward_norm
        
        return raw_reward
    
    def l2r(self, raw_reward: float, t: int = 0) -> float:
        """
        Transform raw reward to final reward with optional ramping.
        
        Args:
            raw_reward: Raw reward value
            t: Current training iteration for ramping
            
        Returns:
            Transformed reward
        """
        if self.reward_exp_ramping > 0:
            reward_exp = 1 + (self.reward_exp - 1) * \
                (1 - 1/(1 + t / self.reward_exp_ramping))
        else:
            reward_exp = self.reward_exp
        
        reward = (raw_reward / self.reward_norm) ** reward_exp
        return reward
    
    def rollout(self, generator, use_rand_policy: bool = True, 
                replay: bool = False) -> List[Tuple]:
        """
        Generate a molecular trajectory with improved structure.
        
        Args:
            generator: Neural network generator
            use_rand_policy: Whether to use random actions
            replay: Whether to add to replay buffer
            
        Returns:
            List of trajectory samples
        """
        m = BlockMoleculeDataExtended()
        samples = []
        trajectory_stats = []
        
        # Determine block limits
        min_blocks, max_blocks = self._determine_block_limits(False)
        
        for t in range(max_blocks):
            # Get action from generator
            action, stats = self._get_action(generator, m, t, min_blocks, use_rand_policy)
            trajectory_stats.append(stats)
            
            # Execute action and get sample
            sample, is_terminal = self._execute_action(
                m, action, t, min_blocks, max_blocks
            )
            
            if sample:
                samples+=sample
                
            if is_terminal:
                break
            
            # Update molecule for next iteration
            if action > 0:
                m = sample[-1][3]  # Get new molecule from sample
        
        # Calculate final statistics
        self._calculate_final_stats(generator, samples, trajectory_stats)
        
        # Add to replay buffer if needed
        if replay:
            self._add_mol_to_replay(samples[-1][3])
        
        return samples
    
    def _determine_block_limits(self, use_rand_policy: bool) -> Tuple[int, int]:
        """Determine min and max blocks for trajectory"""
        min_blocks = self.min_blocks
        max_blocks = self.max_blocks
        
        # Optional: Add randomization for exploration
        if use_rand_policy:
            min_blocks = np.random.randint(1, max_blocks -2)
            exp_blocks = np.random.randint(1, 10**4)
            min_blocks = min_blocks + int(np.log(exp_blocks))
            min_blocks = min(min_blocks, max_blocks - 2)
        
        return min_blocks, max_blocks
    
    def _get_action(self, generator, m: BlockMoleculeDataExtended, 
                    t: int, min_blocks: int, use_rand_policy: bool) -> Tuple[int, Tuple]:
        """
        Get action from generator with proper handling.
        
        Returns:
            Tuple of (action, statistics)
        """
        s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        s_o, m_o = generator(s, do_stems=True)
        
        # Prevent stopping when below minimum blocks
        if t < min_blocks:
            m_o = m_o * 0 - 1000
        
        logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
        
        # Sample action
        if use_rand_policy and self.random_action_prob > 0:
            if self.train_rng.uniform() < self.random_action_prob:
                action = self.train_rng.randint(
                    int(t < min_blocks), logits.shape[0]
                )
            else:
                cat = torch.distributions.Categorical(logits=logits)
                action = cat.sample().item()
        else:
            cat = torch.distributions.Categorical(logits=logits)
            action = cat.sample().item()
        
        # Collect statistics
        q = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
        stats = (q[action].item(), action, torch.logsumexp(q, 0).item())
        
        return action, stats
    
    def _execute_action(self, m: BlockMoleculeDataExtended, action: int, 
                       t: int, min_blocks: int, max_blocks: int) -> Tuple[Optional[Tuple], bool]:
        """
        Execute action and return sample with terminal flag.
        
        Returns:
            Tuple of (sample, is_terminal)
        """
        sample = []
        # Terminal action
        if t >= min_blocks and action == 0:
            r, raw_r = self._get_reward(m)
            sample.append(((m,), ((-1, 0),), r, m, 2))
            return sample, True
        
        # Non-terminal action
        action = max(0, action - 1)
        action_tuple = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
        
        m_old = m
        try:
            m_new = self.mdp.add_block_to(m, *action_tuple)
        except Exception as e:
            self.logger.error(f"Failed to add block: {e}")
            # Return terminal state on error
            r, raw_r = self._get_reward(m_old)
            sample.append(((m_old,), ((-1, 0),), r, m_old, 1))
            return sample, True
        
        # Check if molecule is complete
        if (len(m_new.blocks) and not len(m_new.stems)) or t == max_blocks - 1:
            r, raw_r = self._get_reward(m_new)
            if self.ignore_parents:
                sample.append(((m_old,), (action_tuple,), r, m_new, 1))
            else:
                parents, actions = zip(*self.mdp.parents(m_new))
                sample.append((parents, actions, r, m_new, 1))
            return sample, True
        else:
            if self.ignore_parents:
                sample.append(((m_old,), (action_tuple,), 0, m_new, 0))
            else:
                parents, actions = zip(*self.mdp.parents(m_new))
                sample.append((parents, actions, 0, m_new, 0))
            return sample, False
    
    def _calculate_final_stats(self, generator, samples: List, 
                               trajectory_stats: List):
        """Calculate and store final trajectory statistics"""
        if not samples:
            return
            
        # Get final molecule states
        p = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in samples[-1][0]])
        qp = generator(p)
        
        # Calculate inflow
        qsa_p = generator.model.index_output_by_action(
            p, qp[0], qp[1][:, 0],
            torch.tensor(samples[-1][1], device=self._device).long()
        )
        inflow = torch.logsumexp(qsa_p.flatten(), 0).item()
        
        # Get reward information
        m = samples[-1][3]
        if hasattr(self, '_last_raw_reward'):
            raw_r = self._last_raw_reward
        else:
            _, raw_r = self._get_reward(m)
        
        # Store with thread safety
        mol_data = ([i.cpu().numpy() for i in raw_r], m, trajectory_stats, inflow)
        self._add_sampled_mol(mol_data)
    
    def execute_train_episode_batch(self, generator, dataset=None, 
                                   use_rand_policy: bool = True) -> List:
        """
        Execute a batch of training episodes.
        
        Args:
            generator: Neural network generator
            dataset: Optional dataset for training
            use_rand_policy: Whether to use random policy
            
        Returns:
            List of trajectory samples
        """
        samples = []
        for i in range(self.args.trajectories_mbsize):
            trajectory = self.rollout(generator, use_rand_policy)
            samples.extend(trajectory)
        
        return zip(*samples)
    
    def sample2batch(self, mb: Tuple) -> Tuple:
        """
        Convert samples to batch format for training.
        
        Args:
            mb: Mini-batch of samples
            
        Returns:
            Formatted batch for training
        """
        p, a, r, s, d, *o = mb
        mols = (p, s)
        
        # Create batch indices
        p_batch = torch.tensor(
            sum([[i]*len(p) for i, p in enumerate(p)], []),
            device=self._device
        ).long()
        
        # Convert to representations
        p = self.mdp.mols2batch(list(map(self.mdp.mol2repr, sum(p, ()))))
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        
        # Format actions, rewards, and dones
        a = torch.tensor(sum(a, ()), device=self._device).long()
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)
        
        return (p, p_batch, a, r, s, d, mols, *o)
    
    def start_samplers(self, generator, n: int, dataset) -> callable:
        """
        Start multiple sampling threads with proper management.
        
        Args:
            generator: Neural network generator
            n: Number of sampling threads
            dataset: Training dataset
            
        Returns:
            Function to get samples
        """
        self.ready_events = [threading.Event() for _ in range(n)]
        self.resume_events = [threading.Event() for _ in range(n)]
        self.results = [None] * n
        
        def sampler_thread(idx: int):
            """Thread function for sampling"""
            while not self.stop_event.is_set():
                try:
                    self.results[idx] = self.sample2batch(
                        self.execute_train_episode_batch(
                            generator, dataset, use_rand_policy=True
                        )
                    )
                except Exception as e:
                    self.logger.error(f"Exception in sampler thread {idx}: {e}")
                    self.sampler_threads[idx].failed = True
                    self.sampler_threads[idx].exception = e
                    self.ready_events[idx].set()
                    break
                
                self.ready_events[idx].set()
                self.resume_events[idx].clear()
                self.resume_events[idx].wait()
        
        # Create and start threads
        self.sampler_threads = [
            threading.Thread(target=sampler_thread, args=(i,))
            for i in range(n)
        ]
        
        for thread in self.sampler_threads:
            thread.failed = False
            thread.daemon = True
            thread.start()
        
        # Round-robin sampling
        round_robin_idx = [0]
        
        def get_sample():
            """Get next available sample"""
            while True:
                idx = round_robin_idx[0]
                round_robin_idx[0] = (round_robin_idx[0] + 1) % n
                
                if self.ready_events[idx].is_set():
                    r = self.results[idx]
                    self.ready_events[idx].clear()
                    self.resume_events[idx].set()
                    return r
                elif round_robin_idx[0] == 0:
                    time.sleep(0.001)
        
        return get_sample
    
    def stop_samplers_and_join(self):
        """Stop all sampling threads and wait for completion"""
        self.stop_event.set()
        
        if hasattr(self, 'sampler_threads'):
            # Signal all threads to resume
            for event in self.resume_events:
                event.set()
            
            # Wait for all threads to complete
            for thread in self.sampler_threads:
                thread.join(timeout=1.0)
            
            self.logger.info("All sampling threads stopped")

class RolloutWorker_Leak(RolloutWorker):
    """
    Worker class for generating molecular trajectories and computing rewards.
    Thread-safe implementation with proper memory management.
    """    
    def _get_action(self, generator, m: BlockMoleculeDataExtended, 
                    t: int, min_blocks: int, use_rand_policy: bool) -> Tuple[int, Tuple]:
        """
        Get action from generator with proper handling.
        
        Returns:
            Tuple of (action, statistics)
        """
        s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        (s_o, s_o_full), m_o = generator(s, do_stems=True, return_leaky=True)
        
        # Prevent stopping when below minimum blocks
        if t < min_blocks:
            m_o = m_o * 0 - 1000
        
        logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
        
        # Sample action
        if use_rand_policy and self.train_rng.uniform() < self.random_action_prob:
            logits = torch.cat([m_o.reshape(-1), s_o_full.reshape(-1)])
        cat = torch.distributions.Categorical(logits=logits)
        action = cat.sample().item()
        # Collect statistics
        q = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
        stats = (q[action].item(), action, torch.logsumexp(q, 0).item())
        
        return action, stats
    
    def _execute_action(self, m: BlockMoleculeDataExtended, action: int, 
                       t: int, min_blocks: int, max_blocks: int) -> Tuple[Optional[Tuple], bool]:
        """
        Execute action and return sample with terminal flag.
        
        Returns:
            Tuple of (sample, is_terminal)
        """
        sample = []
        # Terminal action
        if t >= min_blocks and action == 0:
            r, raw_r = self._get_reward(m)
            sample.append(((m,), ((-1, 0),), r, m, 2))
            return sample, True
        
        # Non-terminal action
        action = max(0, action - 1)
        action_tuple = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
        
        m_old = m
        try:
            m_new = self.mdp.add_block_to(m, *action_tuple)
        except Exception as e:
            self.logger.error(f"Failed to add block: {e}")
            # Return terminal state on error
            r, raw_r = self._get_reward(m_old)
            sample.append(((m_old,), ((-1, 0),), r, m_old, 2))
            return sample, True
        
        # Check if molecule is complete
        if (len(m_new.blocks) and not len(m_new.stems)) or t == max_blocks - 1:
            done_type = 2 if len(m_new.stems) == 0 else 1
            r, raw_r = self._get_reward(m_new)
            if self.ignore_parents:
                sample.append(((m_old,), (action_tuple,), r, m_new, done_type))
            else:
                parents, actions = zip(*self.mdp.parents(m_new))
                sample.append((parents, actions, r, m_new, done_type))
            return sample, True
        else:
            if self.ignore_parents:
                sample.append(((m_old,), (action_tuple,), 0, m_new, 0))
            else:
                parents, actions = zip(*self.mdp.parents(m_new))
                sample.append((parents, actions, 0, m_new, 0))
            return sample, False

class RolloutWorker_TB(RolloutWorker):
    """RolloutWorker specialized for Trajectory Balance training."""
    
    def execute_train_episode_batch(self, generator, dataset=None, 
                                   use_rand_policy: bool = True) -> List:
        """
        Execute a batch of training episodes for TB.
        Returns batch with trajectory indices and lengths.
        """
        trajectories = []
        for i in range(self.args.trajectories_mbsize):
            trajectory = self.rollout(generator, use_rand_policy)
            trajectories.append(trajectory)
        batch = (*zip(*sum(trajectories, [])),
                 sum([[i] * len(t) for i, t in enumerate(trajectories)], []),
                 [len(t) for t in trajectories])
        return batch

    def sample2batch(self, mb):
        """Convert samples to batch format for TB training."""
        s, a, r, sp, d, idc, lens = mb
        mols = (s, sp)
        s = self.mdp.mols2batch([self.mdp.mol2repr(i[0]) for i in s])
        a = torch.tensor(sum(a, ()), device=self._device).long()
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)
        # n: number of parents for each state
        n = torch.tensor(
            [len(self.mdp.parents(m)) if (m is not None and aa[0] != -1) else 1 
             for aa, m in zip(sum([list(x) for x in mb[1]], []), sp)], 
            device=self._device
        ).to(self.floatX)
        idc = torch.tensor(idc, device=self._device).long()
        lens = torch.tensor(lens, device=self._device).long()
        # w is placeholder for compatibility
        w = torch.ones(len(r), device=self._device).to(self.floatX)
        return (s, a, w, r, d, n, mols, idc, lens)
    
    def _execute_action(self, m: BlockMoleculeDataExtended, action: int, 
                       t: int, min_blocks: int, max_blocks: int) -> Tuple[Optional[Tuple], bool]:
        """
        Execute action and return sample with terminal flag for TB.
        Sample format: (parents, actions, reward, new_mol, done_flag)
        """
        sample = []
        # Terminal action
        if t >= min_blocks and action == 0:
            r, raw_r = self._get_reward(m)
            sample.append(((m,), ((-1, 0),), r, m, 1))
            return sample, True
        
        # Non-terminal action
        action = max(0, action - 1)
        action_tuple = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
        
        m_old = m
        try:
            m_new = self.mdp.add_block_to(m, *action_tuple)
        except Exception as e:
            self.logger.error(f"Failed to add block: {e}")
            # Return terminal state on error
            r, raw_r = self._get_reward(m_old)
            sample.append(((m_old,), ((-1, 0),), r, m_old, 1))
            return sample, True
        
        # Check if molecule is complete
        if (len(m_new.blocks) and not len(m_new.stems)) or t == max_blocks - 1:
            r, raw_r = self._get_reward(m_new)
            sample.append(((m_old,), (action_tuple,), r, m_new, 1))
            return sample, True
        else:
            sample.append(((m_old,), (action_tuple,), 0, m_new, 0))
            return sample, False

    def rollout(self, generator, use_rand_policy: bool = True, 
                replay: bool = False) -> List[Tuple]:
        """Generate a molecular trajectory for TB training."""
        m = BlockMoleculeDataExtended()
        samples = []
        trajectory_stats = []
        
        # Determine block limits
        min_blocks, max_blocks = self._determine_block_limits(False)
        
        for t in range(max_blocks):
            # Get action from generator
            action, stats = self._get_action(generator, m, t, min_blocks, use_rand_policy)
            trajectory_stats.append(stats)
            
            # Execute action and get sample
            sample, is_terminal = self._execute_action(
                m, action, t, min_blocks, max_blocks
            )
            
            if sample:
                samples += sample
                
            if is_terminal:
                break
            
            # Update molecule for next iteration
            if action > 0:
                m = sample[-1][3]  # Get new molecule from sample
        
        # Calculate final statistics
        self._calculate_final_stats_tb(generator, samples, trajectory_stats)
        
        return samples

    def _calculate_final_stats_tb(self, generator, samples: List, 
                                   trajectory_stats: List):
        """Calculate and store final trajectory statistics for TB."""
        if not samples:
            return
            
        # Get final molecule
        m = samples[-1][3]
        _, raw_r = self._get_reward(m)
        
        # Store with thread safety
        mol_data = ([i.cpu().numpy() for i in raw_r], m, trajectory_stats, 0)
        self._add_sampled_mol(mol_data)


# ============================================================================
# Training Functions
# ============================================================================
class ModelCheckpointer:
    """Handle model checkpointing with versioning"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.checkpoint_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, generator, rollout_worker, iteration: int, 
                       metrics: Optional[Dict] = None):
        """Save a checkpoint with metadata"""
        checkpoint = {
            'iteration': iteration,
            'generator_state': generator.state_dict(),
            'sampled_mols': rollout_worker.sampled_mols[-1000:],  # Save last 1000
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        # Save with iteration number
        path = os.path.join(
            self.checkpoint_dir, 
            f'{Constants.CHECKPOINT_PREFIX}_{iteration}.pth'
        )
        torch.save(checkpoint, path)
        df = pd.DataFrame.from_dict([{'reward':d[0][0], f'score_{rollout_worker.args.objectives[0]}':d[0][1][0], 'traj_len': len(d[2]), 'smiles':d[1].smiles} for d in checkpoint['sampled_mols']])
        df.to_csv(os.path.join(self.log_dir, f'sampled_mols_{iteration}.tsv'), index=False, sep='\t')
        # Also save as latest
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save molecules separately
        mol_path = os.path.join(
            self.log_dir, 
            f'sampled_mols_{iteration}.pkl.gz'
        )
        with gzip.open(mol_path, 'wb') as f:
            pickle.dump(rollout_worker.sampled_mols, f)
        
        return path
    
    def load_checkpoint(self, generator, path: str):
        """Load a checkpoint"""
        checkpoint = torch.load(path)
        generator.load_state_dict(checkpoint['generator_state'])
        return checkpoint

def train_generative_model_with_oracle(
    args, 
    generator, 
    bpath: str, 
    oracle: Oracle, 
    dataset=None, 
    do_save: bool = False
) -> Tuple[RolloutWorker, Dict]:
    """
    Main training loop with improved structure and error handling.
    
    Args:
        args: Training arguments
        generator: Neural network generator
        bpath: Path to molecular building blocks
        oracle: Oracle for scoring
        dataset: Optional training dataset
        do_save: Whether to save checkpoints
        
    Returns:
        Tuple of (rollout_worker, training_metrics)
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")
    
    # Initialize components
    device = args.device
    # Use appropriate RolloutWorker based on criterion
    if args.criterion == 'LeakGFN':
        rollout_worker = RolloutWorker_Leak(args, bpath, oracle, device)
    elif args.criterion in ['TB', 'Reinforce', 'SubTB']:
        rollout_worker = RolloutWorker_TB(args, bpath, oracle, device)
    else:
        rollout_worker = RolloutWorker(args, bpath, oracle, device)
    
    # Load test molecules if available
    if hasattr(args, 'test_mols') and args.test_mols:
        try:
            with gzip.open(args.test_mols, 'rb') as f:
                rollout_worker.test_mols = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load test molecules: {e}")
            rollout_worker.test_mols = []
    
    # Initialize checkpointer
    checkpointer = ModelCheckpointer(args.log_dir) if do_save else None
    
    # Initialize metrics tracking
    metrics_tracker = MetricsTracker()
    
    # Determine if using multi-threading
    multi_thread = not args.debug
    sampler = None
    
    if multi_thread:
        sampler = rollout_worker.start_samplers(
            generator, Constants.THREAD_POOL_SIZE, dataset
        )
    
    # Training loop
    try:
        train_loop(
            args, generator, rollout_worker, oracle, 
            dataset, sampler, checkpointer, metrics_tracker, multi_thread
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        rollout_worker.stop_samplers_and_join()
        
        # Save final checkpoint
        if checkpointer:
            checkpointer.save_checkpoint(
                generator, rollout_worker, 
                args.num_iterations, metrics_tracker.get_summary()
            )
    
    return rollout_worker, metrics_tracker.get_summary()

def train_loop(
    args, generator, rollout_worker, oracle, dataset, 
    sampler, checkpointer, metrics_tracker, multi_thread: bool
):
    """Main training loop with improved structure"""
    logger = logging.getLogger(__name__)
    
    # Get objective threshold
    objective_name = args.objectives[0]
    threshold = Constants.OBJECTIVE_THRESHOLDS.get(objective_name, 0.5)
    
    # Training state
    last_losses = []
    time_last_log = time.time()
    
    for iteration in range(1, args.num_iterations + 1):
        # Get training batch
        if multi_thread:
            batch = sampler()
            # Check for thread failures
            for thread in rollout_worker.sampler_threads:
                if thread.failed:
                    raise RuntimeError(f"Sampler thread failed: {thread.exception}")
        else:
            batch = rollout_worker.sample2batch(
                rollout_worker.execute_train_episode_batch(
                    generator, dataset, use_rand_policy=True
                )
            )
        
        # Training step - different batch formats for different criteria
        if args.criterion in ['TB', 'Reinforce', 'SubTB']:
            # TB/Reinforce/SubTB batch format: (s, a, w, r, d, n, mols, idc, lens)
            s, a, w, r, d, n, mols, idc, lens = batch
            loss = generator.train_step(s, a, w, r, d, n, mols, idc, lens, iteration)
        else:
            # FM batch format: (p, pb, a, r, s, d, mols)
            p, pb, a, r, s, d, mols = batch
            loss = generator.train_step(p, pb, a, r, s, d, mols, iteration)
        
        # Track metrics
        metrics_tracker.add_loss(loss)
        last_losses.append(loss)
        
        # Log to wandb/tensorboard based on criterion
        args.logger.add_object('Loss/train', loss[0], use_context=False)
        if args.criterion == 'LeakGFN':
            # FMGFlowNet returns (loss, term_loss, flow_loss, chem_loss)
            args.logger.add_object('Loss/term', loss[1], use_context=False)
            args.logger.add_object('Loss/flow', loss[2], use_context=False)
            args.logger.add_object('Loss/chem', loss[3], use_context=False)
        elif args.criterion == 'FM':
            # FMGFlowNet returns (loss, term_loss, flow_loss)
            args.logger.add_object('Loss/term', loss[1], use_context=False)
            args.logger.add_object('Loss/flow', loss[2], use_context=False)
        elif args.criterion in ['TB', 'SubTB']:
            # TBGFlowNet/SubTBGFlowNet returns (loss, logZ)
            args.logger.add_object('Loss/logZ', loss[1], use_context=False)
        # MOReinforce returns only (loss,)
        
        # Periodic logging
        if iteration % Constants.LOG_INTERVAL == 0:
            # Calculate average loss (handle different loss formats)
            if last_losses:
                avg_loss = [np.mean([l[i] for l in last_losses if i < len(l)]) 
                           for i in range(len(last_losses[0]))]
            else:
                avg_loss = []
            elapsed = time.time() - time_last_log
            
            # Format log message based on criterion
            if args.criterion == 'LeakGFN':
                loss_str = f'Loss={avg_loss[0]:.4f}, Term={avg_loss[1]:.4f}, Flow={avg_loss[2]:.4f}, chem={avg_loss[3]:.4f}'
            elif args.criterion == 'FM':
                loss_str = f'Loss={avg_loss[0]:.4f}, Term={avg_loss[1]:.4f}, Flow={avg_loss[2]:.4f}'
            elif args.criterion in ['TB', 'SubTB']:
                loss_str = f'Loss={avg_loss[0]:.4f}, logZ={avg_loss[1]:.4f}'
            else:  # Reinforce
                loss_str = f'Loss={avg_loss[0]:.4f}'
            
            logger.info(
                f'Iter {iteration}: {loss_str}, '
                f'Time {elapsed:.2f}s, '
                f'Mols sampled: {len(rollout_worker.sampled_mols)}'
            )
            
            last_losses = []
            time_last_log = time.time()
            # rollout_worker.random_action_prob = rollout_worker.random_action_prob*0.99
        
        # Periodic evaluation
        if iteration % args.sample_iterations == 0:
            eval_metrics = evaluate_model(
                args, generator, rollout_worker, threshold
            )
            
            metrics_tracker.add_eval_metrics(iteration, eval_metrics)
            
            # Save checkpoint if improving
            # if checkpointer and eval_metrics['improved']:
            checkpointer.save_checkpoint(
                generator, rollout_worker, iteration, eval_metrics
            )
        
        # Clean GPU memory periodically
        if iteration % Constants.GPU_MEMORY_CLEANUP_INTERVAL == 0:
            torch.cuda.empty_cache()
        
        # Update logger
        args.logger.update()

def evaluate_model(args, generator, rollout_worker, threshold: float) -> Dict:
    """Evaluate model performance"""
    logger = logging.getLogger(__name__)
    
    # Run evaluation
    top_rewards, diversity, mean_reward, mean_score, rewards = evaluate(
        args, generator, rollout_worker, 100
    )
    
    # Calculate metrics
    rewards_above_threshold = [r for r in rewards if r > threshold]
    success_rate = len(rewards_above_threshold) / len(rewards) if rewards else 0
    
    logger.info(
        f'Evaluation: Top rewards={top_rewards:.3f}, '
        f'Diversity={diversity:.3f}, Mean reward={mean_reward:.3f}, '
        f'Success rate={success_rate:.2%}'
    )
    
    # Log to wandb/tensorboard
    args.logger.add_object('Eval/Top-100-dists', diversity, use_context=False)
    args.logger.add_object('Eval/Top-100-rewards', top_rewards, use_context=False)
    args.logger.add_object('Eval/mean_reward', mean_reward, use_context=False)
    args.logger.add_object('Eval/mean_score', mean_score, use_context=False)
    args.logger.add_object('Eval/success_rate', success_rate, use_context=False)
    
    # Check if this is best so far
    improved = False
    if not hasattr(evaluate_model, 'best_reward'):
        evaluate_model.best_reward = 0
    
    if mean_reward > evaluate_model.best_reward:
        evaluate_model.best_reward = mean_reward
        improved = True
        logger.info(f'New best mean reward: {mean_reward:.3f}')
    
    return {
        'top_rewards': top_rewards,
        'diversity': diversity,
        'mean_reward': mean_reward,
        'mean_score': mean_score,
        'success_rate': success_rate,
        'improved': improved
    }

class MetricsTracker:
    """Track training and evaluation metrics"""
    
    def __init__(self):
        self.train_losses = []
        self.eval_metrics = []
        self.start_time = time.time()
    
    def add_loss(self, loss: List):
        """Add training loss"""
        self.train_losses.append({
            'iteration': len(self.train_losses) + 1,
            'loss': loss,
            'timestamp': time.time() - self.start_time
        })
    
    def add_eval_metrics(self, iteration: int, metrics: Dict):
        """Add evaluation metrics"""
        metrics['iteration'] = iteration
        metrics['timestamp'] = time.time() - self.start_time
        self.eval_metrics.append(metrics)
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        return {
            'train_losses': self.train_losses,
            'eval_metrics': self.eval_metrics,
            'total_time': time.time() - self.start_time
        }

# ============================================================================
# Utility Functions
# ============================================================================
def validate_args(args) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if args.min_blocks >= args.max_blocks:
        raise ValueError("min_blocks must be less than max_blocks")
    
    if args.reward_min > args.reward_norm:
        raise ValueError("reward_min must be less than or equal to reward_norm")
    
    if args.num_iterations < 0:
        raise ValueError("num_iterations must be positive")
    
    if args.trajectories_mbsize < 1:
        raise ValueError("trajectories_mbsize must be at least 1")
    
    return True

def setup_logging(log_dir: str):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

# ============================================================================
# Main Function
# ============================================================================
def main(args):
    """Main entry point with improved error handling"""
    # Validate arguments
    validate_args(args)
    
    # Setup
    set_random_seed(args.seed)
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting training with config: {vars(args)}")
    
    # Set context
    args.logger.set_context('iter_0')
    
    # Initialize Oracle
    from types import SimpleNamespace
    oracle = Oracle(SimpleNamespace(
        objectives=args.objectives, 
        device=args.device
    ))
    
    # Initialize Generator
    if args.criterion == 'LeakGFN':
        generator = LeakFMGFlowNet(args, args.bpath)
    elif args.criterion == 'TB':
        generator = TBGFlowNet(args, args.bpath)
    elif args.criterion == 'FM':
        generator = FMGFlowNet(args, args.bpath)
    elif args.criterion == 'Reinforce':
        generator = MOReinforce(args, args.bpath)
    elif args.criterion == 'SubTB':
        generator = SubTBGFlowNet(args, args.bpath)
    else:
        raise ValueError(f'Unknown criterion: {args.criterion}')
    
    # Set precision
    if args.floatX == 'float64':
        generator = generator.double()
    
    # Move to device
    generator = generator.to(args.device)
    
    # Train model
    try:
        rollout_worker, training_metrics = train_generative_model_with_oracle(
            args, generator, args.bpath, oracle, do_save=args.save
        )
        
        # Save final results
        if args.save:
            args.logger.save(os.path.join(args.log_dir, 'logged_data.pkl.gz'))
            logger.info("Training completed successfully")
        
        return rollout_worker, training_metrics
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

# ============================================================================
# Entry Point
# ============================================================================
if __name__ == '__main__':
    # Parse arguments
    args = argparser()
    
    # Setup directories
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.log_dir = os.path.join(args.log_dir, args.objectives, now)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save configuration
    args_dict = vars(args)
    with open(os.path.join(args.log_dir, "args_info.json"), "w") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)
    
    # Initialize wandb
    if not args.debug:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.objectives}_{now}",
            config=args_dict,
            tags=[args.objectives, args.criterion]
        )
    
    # Setup logging
    args.enable_wandb = not args.debug
    args.enable_tensorboard = False
    args.logger = get_logger(args)
    
    # Process objectives
    args.objectives = args.objectives.split(',')
    
    try:
        # Run training
        main(args)
    finally:
        # Cleanup
        if not args.debug:
            wandb.finish()
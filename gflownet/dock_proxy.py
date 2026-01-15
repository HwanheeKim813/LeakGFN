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

from .mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from .oracle.oracle import Oracle
from .utils import metrics, utils, get_logger, arguments, chem

import model_atom, model_block, model_fingerprint

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


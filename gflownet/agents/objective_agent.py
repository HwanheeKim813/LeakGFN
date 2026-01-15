"""
Objective-specific Agent implementation for GFlowMoA
"""


import os
import yaml
import json
from typing import Dict, Tuple, Any, Optional, List
from types import SimpleNamespace

import torch
import torch.nn as nn
from gflowmoa_1118.generator.gfn import TBGFlowNet, FMGFlowNet
import gflowmoa_1118.generator.gfn as gfn
from gflowmoa_1118.mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
print(f"\n\n\ngfn location: {gfn.__file__}\n\n\n")
class ObjectiveSpecificAgent(nn.Module):
    """
    Agent specialized for a specific objective function.
    
    This agent loads a pre-trained model for a specific objective
    and provides action distributions based on that objective.
    """
    
    def __init__(
        self, 
        json_path: str,
        device: str
    ):
        """
        Initialize an objective-specific agent.
        
        Args:
            json_path: Path to the JSON file containing the arguments
        """
        super().__init__()
        with open(json_path, 'r') as f:
            args = json.load(f)
        args = SimpleNamespace(**args)
        self.args = args
        if args.criterion == 'TB':
            self.model = TBGFlowNet(args, args.bpath)
        elif args.criterion == 'FM':
            self.model = FMGFlowNet(args, args.bpath)
        self.device = device
        self.model.to(self.device)
        self.objective = args.objectives
        self.reward_norm = args.reward_norm
        self.reward_min = args.reward_min
        self.reward_bin = args.reward_bin
        self.reward_exp_ramping = args.reward_exp_ramping
        self.reward_exp = args.reward_exp
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        try:
            if checkpoint_path is not None:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['generator_state'])
                print(f"Successfully loaded checkpoint for {self.objective} from {checkpoint_path}")
            else:
                print(f"No checkpoint found for {self.objective}")
        except Exception as e:
            print(f"Error loading checkpoint for {self.objective}: {str(e)}")
            raise
    
    def forward(
        self,
        state: Any,
        do_stems: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action distribution for the current state.
        
        This is the main method that the conductor will use to get
        each agent's opinion on what action to take.
        
        Args:
            state: Current molecular state (graph representation)
            do_stems: Whether to compute stem outputs
            
        Returns:
            Tuple of (stem_outputs, mol_outputs)
        """
        return self.model(state, do_stems=do_stems)
    
    def get_z(self, mdp):
        m = BlockMoleculeDataExtended()
        s = mdp.mols2batch([mdp.mol2repr(m)])
        outputs = self.model(s, do_stems=True)
        s_o, m_o = outputs
        z = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
        z = z.exp().sum()
        return z

    def get_reward(self, scores):
        if isinstance(scores, dict):
            score = [scores[self.objective]]
        elif isinstance(scores, list):
            score = [s[self.objective] for s in scores]
        else:
            raise ValueError(f"Invalid type of scores: {type(scores)}")
        score = torch.tensor(score).to('cpu')
        raw_reward = score.clone()
        # raw_reward[raw_reward>=self.reward_bin] = 1
        # if self.reward_norm != self.reward_bin:
        #     raw_reward[raw_reward<self.reward_bin and raw_reward>self.reward_norm] = self.reward_norm
        # raw_reward[raw_reward<self.reward_bin] = self.reward_min
        reward = self.l2r(raw_reward.clip(self.reward_min))
        return reward, raw_reward, score

    def l2r(self, raw_reward, t=0):
        if self.reward_exp_ramping > 0:
            reward_exp = 1 + (self.reward_exp - 1) * \
                (1 - 1/(1 + t / self.reward_exp_ramping))
            # when t=0, exp = 1; t->∞, exp = self.reward_exp
        else:
            reward_exp = self.reward_exp

        reward = (raw_reward/self.reward_norm)**reward_exp

        return reward

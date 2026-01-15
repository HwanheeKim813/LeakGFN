"""
Agent Pool for managing multiple objective-specific agents
"""
import torch
import yaml
from typing import Dict, List, Any, Optional, Tuple
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from torch import nn
from torch_geometric import nn as gnn

from .objective_agent import ObjectiveSpecificAgent
from mol_mdp_ext import MolMDPExtended
        
class AgentPool(nn.Module):
    """
    Manages a pool of objective-specific agents for multi-objective optimization.
    
    The AgentPool is responsible for:
    - Loading and initializing all agents
    - Providing unified interface for conductor
    - Managing agent states and outputs
    """
    def __init__(
        self,
        args,
        device: str = 'cuda',
    ):

        """
        Initialize the agent pool.
        
        Args:
            config_path: Path to agent pool configuration file
            objectives: List of objectives to use
            device: Device to run on
            bpath: Path to blocks file for MDP
        """
        super().__init__()
        self.device = device
        self.objectives = args.objectives
        self.agents = nn.ModuleDict()
        # Load configuration
        with open(args.best_agent_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # Initialize agents
        self.args = args
        self.log_reg_c = float(args.log_reg_c)
        self.do_nblocks_reg = False
        self.max_blocks = int(args.max_blocks)
        self.leaf_coef = float(args.leaf_coef)
        self.balanced_loss = bool(args.balanced_loss)
        self.clip_grad = float(args.clip_grad)
        self.learning_rate = float(args.learning_rate)
        self.weight_decay = float(args.weight_decay)
        self.pruned_coef = float(args.pruned_coef)
        self.mdp = MolMDPExtended(args.bpath)
        self.mdp.post_init(args.device, args.repr_type,
                      include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        if args.floatX == 'float64':
            self.mdp.floatX = self.floatX = torch.double
        else:
            self.mdp.floatX = self.floatX = torch.float
        self._initialize_agents()
        self.opt = torch.optim.Adam(self.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        self.training_steps = 0
    def get_agents_z(self):
        agents_z = {}
        all_z = 0
        for objective in self.objectives:
            z = self.agents[objective].get_z(self.mdp)
            agents_z[objective] = z
            all_z += z
        normalized_agents_z = {k: all_z/v for k,v in agents_z.items()}
        return agents_z, normalized_agents_z
    def _initialize_agents(self):
        """Initialize all objective-specific agents."""
        agent_configs = self.config.get('agents', {})
        for objective in self.objectives:
            if objective not in agent_configs:
                raise ValueError(f"No configuration found for objective: {objective}")
                
            agent_config = agent_configs[objective]
            # Create model based on configuration
            model = self._create_model(agent_config)
            
            # Build full checkpoint path
            checkpoint_path = self._build_checkpoint_path(agent_config)
            model.load_checkpoint(checkpoint_path)
            # Create agent
            self.agents[objective] = model
        agents_z, normalized_agents_z = self.get_agents_z()
        print(f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
        print(f"Agents z: {agents_z}")
        print(f"normalized Agents z: {normalized_agents_z}")
    def _create_model(self, agent_config: Dict) -> ObjectiveSpecificAgent:
        """Create model based on agent configuration."""
        # Extract model parameters
        json_path = self._build_json_path(agent_config)
        model = ObjectiveSpecificAgent(
            json_path=json_path,
            device=self.device
        )
        return model
    
    def _build_checkpoint_path(self, agent_config: Dict) -> Optional[str]:
        """Build full checkpoint path from configuration."""
        if 'checkpoint_base_path' in agent_config:
            base_path = agent_config.get('checkpoint_base_path', './checkpoints')
        else:
            base_path = self.config.get('checkpoint_base_path', './checkpoints')
        subdir = agent_config.get('checkpoint_subdir', '')
        filename = agent_config.get('checkpoint_file', '')
        
        if subdir and filename:
            return os.path.join(base_path, subdir, filename)
        return None

    def _build_json_path(self, agent_config: Dict) -> Optional[str]:
        """Build full JSON path from configuration."""
        if 'checkpoint_base_path' in agent_config:
            base_path = agent_config.get('checkpoint_base_path', './checkpoints')
        else:
            base_path = self.config.get('checkpoint_base_path', './checkpoints')
        subdir = agent_config.get('checkpoint_subdir', '')
        filename = agent_config.get('json_file', 'agent_info.json')
        if subdir and filename:
            return os.path.join(base_path, subdir, filename)
        return None
    
    def forward(
        self,
        state: Any,
        do_stems: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get outputs from all agents for a given state.
        
        Args:
            state: Current molecular state
            do_stems: Whether to compute stem outputs
            
        Returns:
            Dictionary mapping objective names to (stem_out, mol_out) tuples
        """
        outputs = {}
        for objective, agent in self.agents.items():
            s = state.clone()
            outputs[objective] = agent.forward(s, do_stems=do_stems)
        mol_out = 0
        stem_out = 0
        _, normalized_agents_z = self.get_agents_z()
        for k, output in outputs.items():
            # mol_out += output[0]
            # stem_out += output[1]
            stem_out += torch.exp(output[0])*normalized_agents_z[k]
            mol_out += torch.exp(output[1])*normalized_agents_z[k]
        outputs['combined'] = (torch.log(stem_out+self.log_reg_c), torch.log(mol_out+self.log_reg_c))
        return outputs
    
    def train_step(self, parents, parent_batch, actions, rewards_dict, states, dones, mols):
        """Train the agent pool."""
        loss, term_loss, flow_loss, pruned_loss, loss_dict = self.FMLoss(parents, parent_batch, actions, rewards_dict, states, dones, mols)
        self.opt.zero_grad()
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.agents.parameters(), self.clip_grad)
        self.opt.step()
        self.training_steps += 1
        return loss_dict
    
    def FMLoss(self, parents, parent_batch, actions, rewards_dict, states, dones, mols):
        _, normalized_agents_z = self.get_agents_z()
        outputs_state = self.forward(states)
        outputs_parent = self.forward(parents)
        loss_dict = {}
        loss = 0
        term_loss = 0
        flow_loss = 0
        pruned_loss = 0
        combined_exp_inflow = 0
        combined_exp_outflow = 0
        reward_combined = 0
        for objective, agent in self.agents.items():
            reward = rewards_dict[objective]
            reward_combined += reward * normalized_agents_z[objective]
            ntransitions = reward.shape[0]
            stem_out_s, mol_out_s = outputs_state[objective]
            stem_out_p, mol_out_p = outputs_parent[objective]
            qsa_p = self.index_output_by_action(
                parents, stem_out_p, mol_out_p[:, 0], actions)
            exp_inflow = (torch.zeros((ntransitions,), device=qsa_p.device, dtype=qsa_p.dtype)
                      .index_add_(0, parent_batch, torch.exp(qsa_p)))  # pb is the parents' batch index
            combined_exp_inflow += exp_inflow
            inflow = torch.log(exp_inflow + agent.model.log_reg_c)
            exp_outflow = self.sum_output(states, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
            outflow_plus_r = torch.log(self.log_reg_c + reward * (dones==2) + exp_outflow * (dones!=2))
            outflow_plus_r_pruned = torch.log(self.log_reg_c + exp_outflow)
            if self.do_nblocks_reg:
                losses = _losses = ((inflow - outflow_plus_r) /
                                (states.nblocks * self.max_blocks)).pow(2)
            else:
                losses = _losses = (inflow - outflow_plus_r).pow(2)
                pruned_loss_i = (outflow_plus_r_pruned[dones==1] - torch.log(self.log_reg_c + reward)[dones==1]).pow(2)
            term_loss_i = (losses * (dones==2)).sum() / ((dones==2).sum() + 1e-20)
            flow_loss_i = (losses * (dones!=2)).sum() / ((dones!=2).sum() + 1e-20)
            pruned_loss_i = pruned_loss_i.sum() / ((dones==2).sum() + 1e-20)
            if self.balanced_loss:
                loss_i = term_loss_i * self.leaf_coef + flow_loss_i + pruned_loss_i * self.pruned_coef
            else:
                loss_i = losses.mean()
            term_loss += term_loss_i
            flow_loss += flow_loss_i
            pruned_loss += pruned_loss_i
            loss += loss_i
            loss_dict[objective] = (loss_i.item(), term_loss_i.item(), flow_loss_i.item(), pruned_loss_i.item())
        objective = 'combined'
        reward = reward_combined
        ntransitions = reward.shape[0]
        stem_out_s, mol_out_s = outputs_state[objective]
        stem_out_p, mol_out_p = outputs_parent[objective]
        qsa_p = self.index_output_by_action(
            parents, stem_out_p, mol_out_p[:, 0], actions)
        exp_inflow = (torch.zeros((ntransitions,), device=qsa_p.device, dtype=qsa_p.dtype)
                    .index_add_(0, parent_batch, torch.exp(qsa_p)))  # pb is the parents' batch index
        combined_inflow = torch.log(exp_inflow + self.log_reg_c)
        combined_exp_outflow = self.sum_output(states, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
        combined_outflow_plus_r = torch.log(self.log_reg_c + reward * (dones==2) + combined_exp_outflow * (dones!=2))
        combined_outflow_plus_r_pruned = torch.log(combined_exp_outflow + self.log_reg_c)
        if self.do_nblocks_reg:
            combined_losses = (combined_inflow - combined_outflow_plus_r) / (states.nblocks * self.max_blocks).pow(2)
        else:
            combined_losses = (combined_inflow - combined_outflow_plus_r).pow(2)
            combined_pruned_loss = (combined_outflow_plus_r_pruned[dones==1] - torch.log(self.log_reg_c + reward)[dones==1]).pow(2)
        combined_term_loss = (combined_losses * (dones==2)).sum() / ((dones==2).sum() + 1e-20)
        combined_flow_loss = (combined_losses * (dones!=2)).sum() / ((dones!=2).sum() + 1e-20)
        combined_pruned_loss = combined_pruned_loss.sum() / ((dones==2).sum() + 1e-20)
        if self.balanced_loss:
            combined_loss = combined_term_loss * self.leaf_coef + combined_flow_loss + combined_pruned_loss * self.pruned_coef
        else:
            combined_loss = combined_losses.mean()
        loss_dict['combined'] = (combined_loss.item(), combined_term_loss.item(), combined_flow_loss.item())
        term_loss += combined_term_loss
        flow_loss += combined_flow_loss
        pruned_loss += combined_pruned_loss
        loss += combined_loss
        loss_dict['all'] = (loss.item(), term_loss.item(), flow_loss.item(), pruned_loss.item())
        return loss, term_loss, flow_loss, pruned_loss, loss_dict
    
    
    def get_reward(self, scores):
        """Get reward from all agents."""
        rewards = {}
        combined_reward = 0
        combined_raw_reward = 0
        combined_score = 0
        rewards = {}
        raw_rewards = {}
        for objective, agent in self.agents.items():
            reward, raw_reward, score = agent.get_reward(scores)
            rewards[objective] = reward
            raw_rewards[objective] = (reward, raw_reward, score)
            combined_reward += reward
            combined_raw_reward += raw_reward
            combined_score += score
        rewards['combined'] = combined_reward
        raw_rewards['combined'] = (combined_reward, combined_raw_reward, combined_score)
        return rewards, raw_rewards
    
    def get_agent(self, objective: str) -> ObjectiveSpecificAgent:
        """Get a specific agent by objective name."""
        if objective not in self.agents:
            raise KeyError(f"No agent found for objective: {objective}")
        return self.agents[objective]
    
    def get_agent_by_index(self, index: int) -> ObjectiveSpecificAgent:
        """Get agent by index (order as initialized)."""
        objective = self.objectives[index]
        return self.agents[objective]
    
    def get_objective_by_index(self, index: int) -> str:
        """Get objective name by index."""
        return self.objectives[index]
    
    def out_to_policy(self, s, stem_o, mol_o):
        if self.categorical_style == 'softmax':
            stem_e = torch.exp(stem_o)
            mol_e = torch.exp(mol_o[:, 0])
        elif self.categorical_style == 'escort':
            stem_e = abs(stem_o)**self.escort_p
            mol_e = abs(mol_o[:, 0])**self.escort_p
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e + 1e-8
        return mol_e / Z, stem_e / Z[s.stems_batch, None]

    def action_negloglikelihood(self, s, a, stem_o, mol_o):
        """
        calculate logp
        """
        mol_p, stem_p = self.out_to_policy(s, stem_o, mol_o)
        #print(Z.shape, Z.min().item(), Z.mean().item(), Z.max().item())
        mol_lsm = torch.log(mol_p + 1e-20)
        stem_lsm = torch.log(stem_p + 1e-20)
        #print(mol_lsm.shape, mol_lsm.min().item(), mol_lsm.mean().item(), mol_lsm.max().item())
        #print(stem_lsm.shape, stem_lsm.min().item(), stem_lsm.mean().item(), stem_lsm.max().item(), '--')
        return -self.index_output_by_action(s, stem_lsm, mol_lsm, a)

    def index_output_by_action(self, s, stem_o, mol_o, a):
        stem_slices = torch.tensor(
            s._slice_dict['stems'][:-1], dtype=torch.long, device=stem_o.device)
        return (
            stem_o[stem_slices + a[:, 1]][
                torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            + mol_o * (a[:, 0] == -1))

    def sum_output(self, s, stem_o, mol_o):
        return gnn.global_add_pool(stem_o, s.stems_batch).sum(1) + mol_o

    @property
    def num_agents(self) -> int:
        """Number of agents in the pool."""
        return len(self.agents)
    
    @property
    def num_objectives(self) -> int:
        """Number of objectives (same as num_agents)."""
        return len(self.objectives)
    
    def __repr__(self) -> str:
        return f"AgentPool(objectives={self.objectives})"
    
    def __len__(self) -> int:
        return len(self.agents)

class AgentPool2(AgentPool):
    def forward(
        self,
        state: Any,
        do_stems: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get outputs from all agents for a given state.
        
        Args:
            state: Current molecular state
            do_stems: Whether to compute stem outputs
            
        Returns:
            Dictionary mapping objective names to (stem_out, mol_out) tuples
        """
        outputs = {}
        for objective, agent in self.agents.items():
            s = state.clone()
            outputs[objective] = agent.forward(s, do_stems=do_stems)
        mol_out = 0
        stem_out = 0
        for k, output in outputs.items():
            # mol_out += output[0]
            # stem_out += output[1]
            stem_out += torch.exp(output[0])
            mol_out += torch.exp(output[1])
        outputs['combined'] = (torch.log(stem_out+self.log_reg_c), torch.log(mol_out+self.log_reg_c))
        return outputs
        
class HarmonicAgentPool(AgentPool):
    def get_reward(self, scores):
        """Get reward from all agents."""
        rewards = {}
        combined_reward = 0
        combined_raw_reward = 0
        combined_score = 0
        rewards = {}
        raw_rewards = {}
        for objective, agent in self.agents.items():
            reward, raw_reward, score = agent.get_reward(scores)
            rewards[objective] = reward
            raw_rewards[objective] = (reward, raw_reward, score)
            combined_reward += 1/torch.clamp(reward, min=1e-20)
            combined_raw_reward += 1/torch.clamp(raw_reward, min=1e-20)
            combined_score += 1/torch.clamp(score, min=1e-20)
        rewards['combined'] = 1/torch.clamp(combined_reward, min=1e-20)
        raw_rewards['combined'] = (1/torch.clamp(combined_reward, min=1e-20), 1/torch.clamp(combined_raw_reward, min=1e-20), 1/torch.clamp(combined_score, min=1e-20))
        return rewards, raw_rewards
    def FMLoss(self, parents, parent_batch, actions, rewards_dict, states, dones, mols):
        outputs_state = self.forward(states)
        outputs_parent = self.forward(parents)
        loss_dict = {}
        loss = 0
        term_loss = 0
        flow_loss = 0
        combined_exp_inflow = 0
        combined_exp_outflow = 0
        for objective, agent in self.agents.items():
            reward = rewards_dict[objective]
            ntransitions = reward.shape[0]
            stem_out_s, mol_out_s = outputs_state[objective]
            stem_out_p, mol_out_p = outputs_parent[objective]
            qsa_p = self.index_output_by_action(
                parents, stem_out_p, mol_out_p[:, 0], actions)
            exp_inflow = (torch.zeros((ntransitions,), device=qsa_p.device, dtype=qsa_p.dtype)
                      .index_add_(0, parent_batch, torch.exp(qsa_p)))  # pb is the parents' batch index
            combined_exp_inflow += 1/torch.clamp(exp_inflow, min=1e-20)
            inflow = torch.log(exp_inflow + agent.model.log_reg_c)
            exp_outflow = self.sum_output(states, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
            combined_exp_outflow += 1/torch.clamp(exp_outflow, min=1e-20)
            outflow_plus_r = torch.log(agent.model.log_reg_c + reward + exp_outflow * (1-dones))
            if self.do_nblocks_reg:
                losses = _losses = ((inflow - outflow_plus_r) /
                                (states.nblocks * self.max_blocks)).pow(2)
            else:
                losses = _losses = (inflow - outflow_plus_r).pow(2)
            term_loss_i = (losses * dones).sum() / (dones.sum() + 1e-20)
            flow_loss_i = (losses * (1-dones)).sum() / ((1-dones).sum() + 1e-20)
            if self.balanced_loss:
                loss_i = term_loss_i * self.leaf_coef + flow_loss_i
            else:
                loss_i = losses.mean()
            term_loss += term_loss_i
            flow_loss += flow_loss_i
            loss += loss_i
            loss_dict[objective] = (loss_i.item(), term_loss_i.item(), flow_loss_i.item())
        
        reward = rewards_dict['combined']
        ntransitions = reward.shape[0]
        harmonic_inflow = 1/torch.clamp(combined_exp_inflow, min=1e-20)
        combined_inflow = torch.log(harmonic_inflow + self.log_reg_c)
        harmonic_outflow = 1/torch.clamp(combined_exp_outflow, min=1e-20)
        combined_outflow_plus_r = torch.log(harmonic_outflow + reward + self.log_reg_c)
        if self.do_nblocks_reg:
            combined_losses = (combined_inflow - combined_outflow_plus_r) / (states.nblocks * self.max_blocks).pow(2)
        else:
            combined_losses = (combined_inflow - combined_outflow_plus_r).pow(2)
        combined_term_loss = (combined_losses * dones).sum() / (dones.sum() + 1e-20)
        combined_flow_loss = (combined_losses * (1-dones)).sum() / ((1-dones).sum() + 1e-20)
        if self.balanced_loss:
            combined_loss = combined_term_loss * self.leaf_coef + combined_flow_loss
        else:
            combined_loss = combined_losses.mean()
        loss_dict['combined'] = (combined_loss.item(), combined_term_loss.item(), combined_flow_loss.item())
        term_loss += combined_term_loss
        flow_loss += combined_flow_loss
        loss += combined_loss
        loss_dict['all'] = (loss.item(), term_loss.item(), flow_loss.item())
        return loss, term_loss, flow_loss, loss_dict



class IndividualAgentPool(AgentPool):
    def FMLoss(self, parents, parent_batch, actions, rewards_dict, states, dones, mols):
        outputs_state = self.forward(states)
        outputs_parent = self.forward(parents)
        loss_dict = {}
        loss = 0
        term_loss = 0
        flow_loss = 0
        combined_exp_inflow = 0
        combined_exp_outflow = 0
        for objective, agent in self.agents.items():
            reward = rewards_dict[objective]
            ntransitions = reward.shape[0]
            stem_out_s, mol_out_s = outputs_state[objective]
            stem_out_p, mol_out_p = outputs_parent[objective]
            qsa_p = self.index_output_by_action(
                parents, stem_out_p, mol_out_p[:, 0], actions)
            exp_inflow = (torch.zeros((ntransitions,), device=qsa_p.device, dtype=qsa_p.dtype)
                      .index_add_(0, parent_batch, torch.exp(qsa_p)))  # pb is the parents' batch index
            combined_exp_inflow += exp_inflow
            inflow = torch.log(exp_inflow + agent.model.log_reg_c)
            exp_outflow = self.sum_output(states, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
            combined_exp_outflow += exp_outflow
            outflow_plus_r = torch.log(agent.model.log_reg_c + reward + exp_outflow * (1-dones))
            if self.do_nblocks_reg:
                losses = _losses = ((inflow - outflow_plus_r) /
                                (states.nblocks * self.max_blocks)).pow(2)
            else:
                losses = _losses = (inflow - outflow_plus_r).pow(2)
            term_loss_i = (losses * dones).sum() / (dones.sum() + 1e-20)
            flow_loss_i = (losses * (1-dones)).sum() / ((1-dones).sum() + 1e-20)
            if self.balanced_loss:
                loss_i = term_loss_i * self.leaf_coef + flow_loss_i
            else:
                loss_i = losses.mean()
            term_loss += term_loss_i
            flow_loss += flow_loss_i
            loss += loss_i
            loss_dict[objective] = (loss_i.item(), term_loss_i.item(), flow_loss_i.item())
        
        reward = rewards_dict['combined']
        ntransitions = reward.shape[0]
        combined_inflow = torch.log(combined_exp_inflow + self.log_reg_c)
        combined_outflow_plus_r = torch.log(combined_exp_outflow + reward + self.log_reg_c)
        if self.do_nblocks_reg:
            combined_losses = (combined_inflow - combined_outflow_plus_r) / (states.nblocks * self.max_blocks).pow(2)
        else:
            combined_losses = (combined_inflow - combined_outflow_plus_r).pow(2)
        combined_term_loss = (combined_losses * dones).sum() / (dones.sum() + 1e-20)
        combined_flow_loss = (combined_losses * (1-dones)).sum() / ((1-dones).sum() + 1e-20)
        if self.balanced_loss:
            combined_loss = combined_term_loss * self.leaf_coef + combined_flow_loss
        else:
            combined_loss = combined_losses.mean()
        loss_dict['combined'] = (combined_loss.item(), combined_term_loss.item(), combined_flow_loss.item())
        term_loss += combined_term_loss
        flow_loss += combined_flow_loss
        # loss += combined_loss
        loss_dict['all'] = (loss.item(), term_loss.item(), flow_loss.item())
        return loss, term_loss, flow_loss, loss_dict
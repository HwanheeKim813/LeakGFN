import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging import critical
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_block
import model_atom
import torch_geometric
from torch_scatter import scatter
from mol_mdp_ext import MolMDPExtended
from torch_geometric.nn import global_add_pool, global_mean_pool

def make_model(args, mdp, is_proxy=False):
    repr_type = args.proxy_repr_type if is_proxy else args.repr_type
    nemb = args.proxy_nemb if is_proxy else args.nemb
    num_conv_steps = args.proxy_num_conv_steps if is_proxy else args.num_conv_steps
    model_version = args.proxy_model_version if is_proxy else args.model_version
        
    if repr_type == 'block_graph':
        model = model_block.GraphAgent(nemb=nemb,
                                    nvec=len(args.objectives),
                                    out_per_stem=mdp.num_blocks,
                                    out_per_mol=1,
                                    num_conv_steps=num_conv_steps,
                                    mdp_cfg=mdp,
                                    version='v4',
                                    partition_init=args.partition_init)
    elif repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=1,
                                     num_conv_steps=num_conv_steps,
                                     version=model_version,
                                     do_nblocks=(hasattr(args,'include_nblocks')
                                                 and args.include_nblocks), dropout_rate=0.1)
    elif repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        # model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)

    model.to(args.device)
    if args.floatX == 'float64':
        model = model.double()

    return model

class FMGFlowNet(nn.Module):
    def __init__(self, args, bpath):
        super().__init__()
        self.args = args
        mdp = MolMDPExtended(bpath)
        mdp.post_init(args.device, args.repr_type,
                      include_nblocks=args.include_nblocks)
        mdp.build_translation_table()
        self.model = make_model(args, mdp, is_proxy=False)
        self.opt = torch.optim.Adam(self.model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

        self.loginf = 1000  # to prevent nans
        self.log_reg_c = args.log_reg_c
        self.balanced_loss = args.balanced_loss
        self.do_nblocks_reg = False
        self.max_blocks = args.max_blocks
        self.leaf_coef = args.leaf_coef
        self.pruned_coef = args.pruned_coef
        self.clip_grad = args.clip_grad
        # self.score_criterion = nn.MSELoss(reduction='none')
        self.score_criterion = nn.MSELoss()

    def forward(self, graph_data, vec_data=None, do_stems=True, return_leaky=False):
        return self.model(graph_data, vec_data, do_stems, return_leaky)

    def train_step(self, p, pb, a, r, s, d, mols, i):
        loss, term_loss, flow_loss = self.FMLoss(p, pb, a, r, s, d)

        self.opt.zero_grad()
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad)
        self.opt.step()
        self.model.training_steps = i+1
        
        return (loss.item(), term_loss.item(), flow_loss.item())

    def FMLoss(self, p, pb, a, r, s, d):
        # Since we sampled 'mbsize' trajectories, we're going to get
        # roughly mbsize * H (H is variable) transitions
        ntransitions = r.shape[0]
        # state outputs
        (stem_out_s, stem_out_s_full), mol_out_s = self.model(s, return_leaky=True) # log(F)
        # parents of the state outputs
        (stem_out_p, stem_out_p_full), mol_out_p = self.model(p, return_leaky=True)
        # index parents by their corresponding actions
        qsa_p = self.model.index_output_by_action(
            p, stem_out_p, mol_out_p[:, 0], a)
        # then sum the parents' contribution, this is the inflow
        exp_inflow = (torch.zeros((ntransitions,), device=qsa_p.device, dtype=qsa_p.dtype)
                      .index_add_(0, pb, torch.exp(qsa_p)))  # pb is the parents' batch index
        inflow = torch.log(exp_inflow + self.log_reg_c)
        exp_outflow = self.model.sum_output(s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
        outflow_plus_r = torch.log(self.log_reg_c + r * (d!=0) + exp_outflow * (d==0))

        if self.do_nblocks_reg:
            losses = _losses = ((inflow - outflow_plus_r) /
                                (s.nblocks * self.max_blocks)).pow(2)
        else:
            losses = _losses = (inflow - outflow_plus_r).pow(2)
            # pruned_loss += (torch.exp(mol_out_s[:, 0][d==1]) - torch.log(self.log_reg_c + r)[d==1]).pow(2)
            # pruned_loss = (outflow_plus_r_pruned-torch.log(torch.tensor(self.log_reg_c))).pow(2)
            # pruned_losses = outflow_plus_r_pruned.pow(2)

        term_loss = (losses * (d!=0)).sum() / ((d!=0).sum() + 1e-20)  # terminal nodes
        flow_loss = (losses * (d==0)).sum() / ((d==0).sum() + 1e-20)  # non-terminal nodes

        if self.balanced_loss:
            loss = term_loss * self.leaf_coef + flow_loss
        else:
            loss = losses.mean()

        return loss, term_loss, flow_loss



class TBGFlowNet(nn.Module):
    def __init__(self, args, bpath):
        super().__init__()
        self.args = args
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(args.device, args.repr_type,
                           include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        self.model = make_model(args, self.mdp, is_proxy=False)
        
        # TB-specific: learnable logZ parameter
        self.logZ = torch.nn.Parameter(torch.tensor([5.0], device=args.device))
        self.logZ_lower = 0.0
        
        # Optimizer includes logZ parameter with separate learning rate for Z
        z_lr = getattr(args, 'Z_learning_rate', args.learning_rate)
        self.opt = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': args.learning_rate},
            {'params': [self.logZ], 'lr': z_lr}
        ], weight_decay=args.weight_decay)

    def forward(self, graph_data, vec_data=None, do_stems=True, return_leaky=False):
        return self.model(graph_data, vec_data, do_stems, return_leaky)
    
    def clamp_logZ(self):
        """Clamp logZ to minimum value to prevent collapse."""
        self.logZ.data = torch.clamp(self.logZ, min=self.logZ_lower)

    def train_step(self, s, a, w, r, d, n, mols, idc, lens, i):
        """
        Train step for TB GFlowNet.
        
        Args:
            s: State batch (graph data)
            a: Actions
            w: Weights (not used in TB, for compatibility)
            r: Rewards
            d: Done flags (1 = terminal)
            n: Number of parents for each state
            mols: Molecule data
            idc: Trajectory indices
            lens: Trajectory lengths
            i: Iteration number
        """
        loss = self.TBLoss(s, a, r, d, n, idc, lens)
        self.opt.zero_grad()
        loss.backward()
        if self.args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.clip_grad)
        self.opt.step()
        self.clamp_logZ()
        return (loss.item(), self.logZ.item())

    def TBLoss(self, s, a, r, d, n, idc, lens):
        """
        Trajectory Balance Loss:
        L = (log Z + log P_F(τ) - log R(x) - log P_B(τ))²
        
        Using uniform backward policy: P_B(s'→s) = 1 / |Parents(s')|
        
        Args:
            s: State batch (graph data)
            a: Actions
            r: Rewards
            d: Done flags (1 = terminal)
            n: Number of parents for each state
            idc: Trajectory indices for each transition
            lens: Length of each trajectory
        """
        # Forward policy log probability
        stem_out_s, mol_out_s = self.model(s)
        forward_logp = -self.model.action_negloglikelihood(
            s, a, stem_out_s, mol_out_s)

        # Forward log probability per trajectory: Σ log P_F(a|s)
        forward_ll = scatter(forward_logp, idc, reduce='sum')
        
        # Backward log probability (uniform policy): log P_B = -log(|Parents|)
        log_n_parents = torch.log(n + 1e-10)
        backward_ll = -scatter(log_n_parents, idc, reduce='sum')
        
        # Terminal rewards (only for completed trajectories, d == 1)
        terminal_mask = (d == 1)
        rewards = r[terminal_mask]
        log_rewards = torch.log(rewards + 1e-10)
        
        # TB Loss: (log Z + log P_F(τ) - log R(x) - log P_B(τ))²
        losses = (self.logZ + forward_ll - log_rewards - backward_ll).pow(2)
        
        # Clamp to prevent gradient explosion
        losses = torch.clamp(losses, max=5000)
        loss = losses.mean()

        return loss
        

class SubTBGFlowNet(nn.Module):
    """
    Sub-Trajectory Balance GFlowNet.
    
    SubTB applies balance conditions on all sub-trajectories with geometric weighting.
    - λ = 0: Equivalent to Detailed Balance (DB)
    - λ = 1: Equivalent to Trajectory Balance (TB)
    - 0 < λ < 1: Intermediate sub-trajectory balance
    
    Loss for each transition:
    L = (log F(s) + log P_F(s→s') - log F(s') - log P_B(s'→s))²
    
    With geometric weighting across sub-trajectories using parameter λ.
    """
    
    def __init__(self, args, bpath):
        super().__init__()
        self.args = args
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(args.device, args.repr_type,
                           include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        self.model = make_model(args, self.mdp, is_proxy=False)
        
        # SubTB-specific: learnable logZ parameter
        self.logZ = torch.nn.Parameter(torch.tensor([5.0], device=args.device))
        self.logZ_lower = 0.0
        
        # Lambda parameter for sub-trajectory weighting (0=DB, 1=TB)
        self.lamda = getattr(args, 'subtb_lambda', 0.9)
        
        # Optimizer includes logZ parameter with separate learning rate for Z
        z_lr = getattr(args, 'Z_learning_rate', args.learning_rate)
        self.opt = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': args.learning_rate},
            {'params': [self.logZ], 'lr': z_lr}
        ], weight_decay=args.weight_decay)

    def forward(self, graph_data, vec_data=None, do_stems=True, return_leaky=False):
        return self.model(graph_data, vec_data, do_stems, return_leaky)
    
    def clamp_logZ(self):
        """Clamp logZ to minimum value to prevent collapse."""
        self.logZ.data = torch.clamp(self.logZ, min=self.logZ_lower)

    def train_step(self, s, a, w, r, d, n, mols, idc, lens, i):
        """
        Train step for SubTB GFlowNet.
        
        Args:
            s: State batch (graph data)
            a: Actions
            w: Weights (not used, for compatibility)
            r: Rewards
            d: Done flags (1 = terminal)
            n: Number of parents for each state
            mols: Molecule data
            idc: Trajectory indices
            lens: Trajectory lengths
            i: Iteration number
        """
        loss = self.SubTBLoss(s, a, r, d, n, idc, lens)
        self.opt.zero_grad()
        loss.backward()
        if self.args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.clip_grad)
        self.opt.step()
        self.clamp_logZ()
        return (loss.item(), self.logZ.item())

    def SubTBLoss(self, s, a, r, d, n, idc, lens):
        """
        Sub-Trajectory Balance Loss with λ-weighting.
        
        For each sub-trajectory from position i to j:
        L_{i,j} = (log F(s_i) + Σ log P_F - log F(s_j) - Σ log P_B)²
        
        Total loss uses geometric weighting: λ^(j-i-1) * (1-λ) for intermediate
        and λ^(T-1) for the full trajectory.
        
        Args:
            s: State batch (graph data)
            a: Actions
            r: Rewards
            d: Done flags (1 = terminal)
            n: Number of parents for each state
            idc: Trajectory indices for each transition
            lens: Length of each trajectory
        """
        device = idc.device
        num_trajs = lens.shape[0]
        
        # Forward pass to get policy logits and flow predictions
        stem_out_s, mol_out_s = self.model(s)
        
        # Forward policy log probability: log P_F(s → s')
        log_pf = -self.model.action_negloglikelihood(s, a, stem_out_s, mol_out_s)
        
        # Backward policy log probability (uniform): log P_B = -log(|Parents|)
        log_pb = -torch.log(n + 1e-10)
        
        # State flow: log F(s) - use model's mol_out as flow estimate
        # For non-terminal states, mol_out represents log F(s)
        # For terminal states, log F(x) = log R(x)
        log_F = mol_out_s[:, 0]
        
        # Compute cumulative sums for efficient sub-trajectory computation
        # For each trajectory, we need cumsum of (log_pf - log_pb)
        
        total_loss = torch.tensor(0.0, device=device)
        
        # Process each trajectory
        start_idx = 0
        for traj_idx in range(num_trajs):
            traj_len = lens[traj_idx].item()
            end_idx = start_idx + traj_len
            
            if traj_len == 0:
                start_idx = end_idx
                continue
            
            # Get trajectory-specific tensors
            traj_log_pf = log_pf[start_idx:end_idx]
            traj_log_pb = log_pb[start_idx:end_idx]
            traj_log_F = log_F[start_idx:end_idx]
            traj_d = d[start_idx:end_idx]
            traj_r = r[start_idx:end_idx]
            
            # Terminal state log flow is log R(x)
            terminal_mask = (traj_d == 1)
            if terminal_mask.any():
                terminal_idx = terminal_mask.nonzero(as_tuple=True)[0][-1]
                log_R = torch.log(traj_r[terminal_idx] + 1e-10)
            else:
                log_R = torch.tensor(0.0, device=device)
            
            # Compute sub-trajectory losses with λ-weighting
            # For efficiency, we compute weighted combination
            traj_loss = self._compute_subtb_loss_for_trajectory(
                traj_log_pf, traj_log_pb, traj_log_F, log_R, traj_len
            )
            
            total_loss = total_loss + traj_loss
            start_idx = end_idx
        
        # Average over trajectories
        loss = total_loss / max(num_trajs, 1)
        
        # Clamp to prevent gradient explosion
        loss = torch.clamp(loss, max=5000)
        
        return loss
    
    def _compute_subtb_loss_for_trajectory(self, log_pf, log_pb, log_F, log_R, traj_len):
        """
        Compute SubTB loss for a single trajectory using efficient λ-weighting.
        
        Uses the recursive formulation:
        - DB loss at each step: (log F(s) + log P_F - log F(s') - log P_B)²
        - These are combined with geometric weights based on λ
        
        For λ close to 1, this approximates TB loss.
        For λ close to 0, this emphasizes local (DB) consistency.
        """
        if traj_len == 0:
            return torch.tensor(0.0, device=log_pf.device)
        
        lamda = self.lamda
        
        # Compute individual transition errors (DB-style)
        # error_t = log F(s_t) + log P_F(t) - log F(s_{t+1}) - log P_B(t)
        
        # For states 0 to T-1 (non-terminal), log F comes from model
        # For terminal state T, log F(x) = log R(x)
        
        # Construct log_F_next: log F(s_{t+1}) for each transition
        if traj_len > 1:
            log_F_curr = log_F[:-1]  # log F(s_0), ..., log F(s_{T-2})
            log_F_next = log_F[1:]   # log F(s_1), ..., log F(s_{T-1})
            
            # Last transition: log F(s_{T-1}) -> log R(x)
            log_F_curr = torch.cat([log_F_curr, log_F[-1:]])
            log_F_next = torch.cat([log_F_next, log_R.unsqueeze(0)])
        else:
            # Single transition trajectory
            log_F_curr = log_F
            log_F_next = log_R.unsqueeze(0)
        
        # Transition errors
        errors = log_F_curr + log_pf - log_F_next - log_pb
        
        # Apply λ-weighting for sub-trajectory balance
        # Weight for transition at position t in sub-trajectory of length k:
        # w_t,k = λ^(k-1) * (1-λ) for k < T, and λ^(T-1) for full trajectory
        
        # Simplified approach: use weighted sum of squared errors
        # with exponentially decaying weights from the end
        T = traj_len
        
        if T == 1:
            # Single transition: just use squared error
            loss = errors.pow(2).sum()
        else:
            # Compute cumulative errors for sub-trajectories
            # cumsum_errors[i] = Σ_{t=0}^{i} errors[t]
            cumsum_errors = torch.cumsum(errors, dim=0)
            
            # For sub-trajectory (0, j), the total error is cumsum_errors[j-1]
            # Weight: λ^(j-1) * (1-λ) for j < T, λ^(T-1) for j = T
            
            weights = torch.zeros(T, device=errors.device)
            for j in range(T):
                if j < T - 1:
                    weights[j] = (lamda ** j) * (1 - lamda)
                else:
                    weights[j] = lamda ** j
            
            # Normalize weights
            weights = weights / (weights.sum() + 1e-10)
            
            # Compute weighted loss
            # For sub-trajectory ending at position j (0-indexed), error is cumsum_errors[j]
            subtraj_errors = cumsum_errors  # errors for sub-trajectories (0, 1), (0, 2), ..., (0, T)
            
            loss = (weights * subtraj_errors.pow(2)).sum()
        
        # Add logZ regularization: for the full trajectory
        # logZ + Σ log P_F - log R - Σ log P_B should be 0
        full_traj_error = self.logZ + log_pf.sum() - log_R - log_pb.sum()
        loss = loss + full_traj_error.pow(2)
        
        return loss


class MOReinforce(nn.Module):
    """REINFORCE (Policy Gradient) implementation for molecule generation."""
    
    def __init__(self, args, bpath):
        super().__init__()
        self.args = args
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(args.device, args.repr_type,
                           include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        self.model = make_model(args, self.mdp, is_proxy=False)
        self.opt = torch.optim.Adam(self.model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    def forward(self, graph_data, vec_data=None, do_stems=True, return_leaky=False):
        """Forward pass with optional leaky output for compatibility."""
        if return_leaky:
            out = self.model(graph_data, vec_data, do_stems)
            stem_out, mol_out = out
            return (stem_out, stem_out), mol_out
        return self.model(graph_data, vec_data, do_stems)

    def train_step(self, s, a, w, r, d, n, mols, idc, lens, i):
        """
        Train step for REINFORCE.
        
        Args:
            s: State batch (graph data)
            a: Actions
            w: Weights (not used, for compatibility)
            r: Rewards
            d: Done flags (1 = terminal)
            n: Number of parents (not used in REINFORCE)
            mols: Molecule data
            idc: Trajectory indices
            lens: Trajectory lengths
            i: Iteration number
        """
        loss = self.reinforce_loss(s, a, r, d, idc)
        self.opt.zero_grad()
        loss.backward()
        if self.args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.clip_grad)
        self.opt.step()
        return (loss.item(),)

    def reinforce_loss(self, s, a, r, d, idc):
        """
        REINFORCE Loss with baseline:
        L = -Σ log P(a|s) * (R - baseline)
        
        Args:
            s: State batch (graph data)
            a: Actions
            r: Rewards
            d: Done flags (1 = terminal)
            idc: Trajectory indices
        """
        # Forward policy log probability
        stem_out_s, mol_out_s = self.model(s)
        logits = -self.model.action_negloglikelihood(
            s, a, stem_out_s, mol_out_s)

        # Forward log probability per trajectory using trajectory indices
        forward_ll = scatter(logits, idc, reduce='sum')

        # Terminal rewards with baseline (mean reward)
        terminal_mask = (d == 1)
        rewards = r[terminal_mask]
        baseline = rewards.mean()
        
        # REINFORCE loss: -log P(τ) * (R - baseline)
        losses = -forward_ll * (rewards - baseline)
        loss = losses.mean()

        return loss
    



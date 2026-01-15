import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging import critical
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_block_dual as model_block
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

class LeakFMGFlowNet(nn.Module):
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
        loss, term_loss, flow_loss, pruned_loss = self.FMLoss(p, pb, a, r, s, d)

        self.opt.zero_grad()
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad)
        self.opt.step()
        self.model.training_steps = i+1
        
        return (loss.item(), term_loss.item(), flow_loss.item(), pruned_loss.item())

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

        qsa_p_full = self.model.index_output_by_action(
            p, stem_out_p_full, mol_out_p[:, 0], a)
        exp_inflow_full = (torch.zeros((ntransitions,), device=qsa_p_full.device, dtype=qsa_p_full.dtype)
                      .index_add_(0, pb, torch.exp(qsa_p_full)))  # pb is the parents' batch index
        inflow_full = torch.log(exp_inflow_full + self.log_reg_c)
        exp_outflow_full = self.model.sum_output(s, torch.exp(stem_out_s_full), torch.exp(mol_out_s[:, 0]))
        outflow_plus_r_full = torch.log(self.log_reg_c + r * (d==2) + exp_outflow_full * (d!=2))

        if self.do_nblocks_reg:
            losses = _losses = ((inflow - outflow_plus_r) /
                                (s.nblocks * self.max_blocks)).pow(2)
        else:
            l_loss = _losses = (inflow - outflow_plus_r).pow(2)
            f_loss = (inflow_full - outflow_plus_r_full).pow(2)
            # pruned_loss += (torch.exp(mol_out_s[:, 0][d==1]) - torch.log(self.log_reg_c + r)[d==1]).pow(2)
            # pruned_loss = (outflow_plus_r_pruned-torch.log(torch.tensor(self.log_reg_c))).pow(2)
            # pruned_losses = outflow_plus_r_pruned.pow(2)

        term_loss = (l_loss * (d!=0)).sum() / ((d!=0).sum() + 1e-20)  # terminal nodes
        flow_loss = (l_loss * (d==0)).sum() / ((d==0).sum() + 1e-20)  # non-terminal nodes
        full_loss = (f_loss * (d!=2)).sum() / ((d!=2).sum() + 1e-20)  # non-terminal nodes
        if self.balanced_loss:
            # leaky_loss = leaky_term_loss * self.leaf_coef + leaky_flow_loss
            loss = term_loss * self.leaf_coef + flow_loss + full_loss * self.pruned_coef
        else:
            # loss = term_loss + flow_loss + leaky_term_loss + leaky_flow_loss
            # loss += pruned_loss
            loss = term_loss + flow_loss + full_loss

        return loss, term_loss, flow_loss, full_loss

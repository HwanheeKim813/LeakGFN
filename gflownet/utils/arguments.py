import argparse
import os
import yaml

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", 
                        default="default", 
                        help="Path to configuration file")
    args, remaining_args = parser.parse_known_args()
    # Read the configuration file
    if args.config_file == "default":
        config = {}
    else:
        with open(args.config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    parser.add_argument("--best_agent_path", type=str, default='./checkpoints/moa/best_agent.json' if 'best_agent_path' not in config else config['best_agent_path'])
    #* Model Settings
    parser.add_argument('--device', type=str, default='cuda' if 'device' not in config else config['device'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42 if 'seed' not in config else config['seed'],
                       help='Random seed')
    parser.add_argument('--run', type=int, default=0 if 'run' not in config else config['run'],
                       help='Run number')
    parser.add_argument('--save', type=bool,
                        default=False if 'save' not in config else config['save'], help='Save model.')
    parser.add_argument('--debug', type=bool,
                        default=False if 'debug' not in config else config['debug'], help='debug mode, no multi thread')
    parser.add_argument("--enable_tensorboard", type=bool, 
                        default=False if 'enable_tensorboard' not in config else config['enable_tensorboard'])
    parser.add_argument("--enable_wandb", type=bool, 
                        default=True if 'enable_wandb' not in config else config['enable_wandb'])
    parser.add_argument("--log_dir", default='./checkpoints/soo' if 'log_dir' not in config else config['log_dir'])
    parser.add_argument("--include_nblocks", default=False if 'include_nblocks' not in config else config['include_nblocks'])
    parser.add_argument("--num_samples", default=1000 if 'num_samples' not in config else config['num_samples'], type=int)
    parser.add_argument("--floatX", default='float32' if 'floatX' not in config else config['floatX'])
    parser.add_argument('--sample_iterations', type=int, default=1000 if 'sample_iterations' not in config else config['sample_iterations'], help='sample mols and compute metrics')
    parser.add_argument('--wandb_project', type=str, default='GFlowMoA' if 'wandb_project' not in config else config['wandb_project'])
    parser.add_argument('--bpath', type=str, default='./gflownet/data/blocks_105.json' if 'bpath' not in config else config['bpath'])
    parser.add_argument('--test_mols', type=str, default='./gflownet/data/test_mols_6062.pkl.gz' if 'test_mols' not in config else config['test_mols'])
    # objectives
    parser.add_argument("--objectives", type=str, default='gsk3b' if 'objectives' not in config else config['objectives'])
    # parser.add_argument("--alpha", default=1., type=float,
    #                     help='dirichlet distribution')
    # parser.add_argument("--alpha_vector", default='1,1', type=str)
    
    # GFlowNet
    parser.add_argument("--min_blocks", default=2 if 'min_blocks' not in config else config['min_blocks'], type=int)
    parser.add_argument("--max_blocks", default=8 if 'max_blocks' not in config else config['max_blocks'], type=int)
    parser.add_argument("--num_iterations", default=30000 if 'num_iterations' not in config else config['num_iterations'], type=int)  # 30k
    parser.add_argument("--criterion", default="FM" if 'criterion' not in config else config['criterion'], type=str)
    parser.add_argument("--learning_rate", default=5e-4 if 'learning_rate' not in config else config['learning_rate'],
                        help="Learning rate", type=float)
    parser.add_argument("--Z_learning_rate", default=5e-3 if 'Z_learning_rate' not in config else config['Z_learning_rate'],
                        help="Learning rate", type=float)
    parser.add_argument("--clip_grad", default=0 if 'clip_grad' not in config else config['clip_grad'], type=float)
    parser.add_argument("--trajectories_mbsize", default=16 if 'trajectories_mbsize' not in config else config['trajectories_mbsize'], type=int)
    parser.add_argument("--offline_mbsize", default=0 if 'offline_mbsize' not in config else config['offline_mbsize'], type=int)
    parser.add_argument("--hindsight_mbsize", default=0 if 'hindsight_mbsize' not in config else config['hindsight_mbsize'], type=int)
    parser.add_argument("--reward_min", default=1e-2 if 'reward_min' not in config else config['reward_min'], type=float)
    parser.add_argument("--reward_bin", default=0.5 if 'reward_bin' not in config else config['reward_bin'], type=float)
    parser.add_argument("--reward_norm", default=0.8 if 'reward_norm' not in config else config['reward_norm'], type=float)
    parser.add_argument("--reward_exp", default=6 if 'reward_exp' not in config else config['reward_exp'], type=float)
    parser.add_argument("--reward_exp_ramping", default=0 if 'reward_exp_ramping' not in config else config['reward_exp_ramping'], type=float)
    # Hyperparameters for TB
    parser.add_argument("--partition_init", default=30 if 'partition_init' not in config else config['partition_init'], type=float)
    parser.add_argument("--subtb_lambda", default=0.0 if 'subtb_lambda' not in config else config['subtb_lambda'], type=float)
    # Hyperparameters for FM
    parser.add_argument("--log_reg_c", default=(0.1/8) ** 4 if 'log_reg_c' not in config else config['log_reg_c'], type=float)  # (0.1/8)**8  
    parser.add_argument("--balanced_loss", default=True if 'balanced_loss' not in config else config['balanced_loss'], type=bool)
    parser.add_argument("--leaf_coef", default=10 if 'leaf_coef' not in config else config['leaf_coef'], type=float)
    parser.add_argument("--pruned_coef", default=10 if 'pruned_coef' not in config else config['pruned_coef'], type=float)
    # Architecture
    parser.add_argument("--repr_type", default='block_graph' if 'repr_type' not in config else config['repr_type'], type=str)
    parser.add_argument("--model_version", default='v4' if 'model_version' not in config else config['model_version'], type=str)
    parser.add_argument("--num_conv_steps", default=10 if 'num_conv_steps' not in config else config['num_conv_steps'], type=int)
    parser.add_argument("--nemb", default=256 if 'nemb' not in config else config['nemb'], help="#hidden", type=int)
    parser.add_argument("--weight_decay", default=0 if 'weight_decay' not in config else config['weight_decay'], type=float)
    parser.add_argument("--random_action_prob", default=0.05 if 'random_action_prob' not in config else config['random_action_prob'], type=float)
    parser.add_argument("--bootstrap_tau", default=0 if 'bootstrap_tau' not in config else config['bootstrap_tau'], type=float)
    parser.add_argument("--ray_hidden_dim", default=100 if 'ray_hidden_dim' not in config else config['ray_hidden_dim'], type=int)
    parser.add_argument("--logit_clipping", default=0. if 'logit_clipping' not in config else config['logit_clipping'], type=float)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS
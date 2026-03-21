import argparse

def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # Mode and Exp. Settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="er",
        help="Select CIL method",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="[mnist, cifar10, cifar100, imagenet]",
    )
    parser.add_argument("--n_tasks", type=int, default=5, help="The number of tasks")
    parser.add_argument("--n", type=int, default=50, help="The percentage of disjoint split. Disjoint=100, Blurry=0")
    parser.add_argument("--m", type=int, default=10, help="The percentage of blurry samples in blurry split. Uniform split=100, Disjoint=0")
    parser.add_argument("--rnd_NM", action='store_true', default=False, help="if True, N and M are randomly mixed over tasks.")
    parser.add_argument("--rnd_seed", type=int, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    # Dataset
    parser.add_argument(
        "--log_path",
        type=str,
        default="results",
        help="The path logs are saved.",
    )
    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet18", help="Model name"
    )

    # Train
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")

    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")

    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--init_model",
        action="store_true",
        help="Initilize model parameters for every iterations",
    )
    parser.add_argument(
        "--init_opt",
        action="store_true",
        help="Initilize optimizer states for every iterations",
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="set k when we want to set topk accuracy"
    )

    parser.add_argument(
        "--use_amp", action="store_true", help="Use automatic mixed precision."
    )

    # Transforms
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=['cutmix', 'autoaug'],
        help="Additional train transforms [cutmix, cutout, randaug]",
    )

    parser.add_argument("--gpu_transform", action="store_true", help="perform data transform on gpu (for faster AutoAug).")

    # Regularization
    parser.add_argument(
        "--reg_coef",
        type=int,
        default=100,
        help="weighting for the regularization loss term",
    )

    parser.add_argument("--data_dir", type=str, help="location of the dataset")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")
    # Note
    parser.add_argument("--note", type=str, help="Short description of the exp")

    # Eval period
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")

    parser.add_argument("--temp_batchsize", type=int, help="temporary batch size, for true online")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    # GDumb
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs, for GDumb eval')
    parser.add_argument('--workers_per_gpu', type=int, default=1, help='number of workers per GPU, for GDumb eval')

    # CLIB
    parser.add_argument("--imp_update_period", type=int, default=1,
                        help="period between importance update, in units of model updates (increase for heavy datasets like ImageNet)")
    parser.add_argument('--lr_step', type=float, default=0.95, help='step of iterating lr for adaptive LR')
    parser.add_argument('--lr_length', type=int, default=10, help='period of iterating lr for adaptive LR')
    parser.add_argument('--lr_period', type=int, default=10, help='period of iterating lr for adaptive LR')

    # RM & GDumb
    parser.add_argument("--memory_epoch", type=int, default=256, help="number of training epochs after task for Rainbow Memory")

    # BiC
    parser.add_argument("--distilling", type=bool, default=True, help="use distillation for BiC.")

    # AGEM
    parser.add_argument('--agem_batch', type=int, default=240, help='A-GEM batch size for calculating gradient')

    # MIR
    parser.add_argument('--mir_cands', type=int, default=50, help='# candidates to use for MIR')

    # Prompt-based (ViT)
    # Ours
    parser.add_argument('--use_mask', action='store_true', help='use mask for our method')
    parser.add_argument('--use_contrastiv', action='store_true', help='use contrastive loss for our method')
    parser.add_argument('--use_last_layer', action='store_true', help='use last layer for our method')
    # parser.add_argument('--use_dyna_exp', action='store_true', help='use dynamic expand for our method')
    
    parser.add_argument('--use_afs', action='store_true', help='enable Adaptive Feature Scaling (AFS) in ours')
    parser.add_argument('--use_gsf', action='store_true', help='enable Minor-Class Reinforcement (MCR) in ours')
    
    parser.add_argument('--selection_size', type=int, default=1, help='# candidates to use for ViT_Prompt')
    parser.add_argument('--alpha', type=float, default=0.5, help='# candidates to use for STR hyperparameter')
    parser.add_argument('--gamma', type=float, default=2., help='# candidates to use for STR hyperparameter')
    parser.add_argument('--margin', type=float, default=0.5, help='# candidates to use for STR hyperparameter')

    parser.add_argument('--profile', action='store_true', help='enable profiling for ViT_Prompt')

    # LaPrompt
    parser.add_argument('--laprompt_use_ca', action='store_true', help='enable classifier alignment stage for laprompt mode')
    parser.add_argument('--laprompt_ca_lr', type=float, default=0.01, help='classifier alignment learning rate for laprompt mode')
    parser.add_argument('--laprompt_ca_epochs', type=int, default=5, help='classifier alignment epochs for laprompt mode')
    parser.add_argument('--laprompt_ca_storage', type=str, default='variance', help='statistics type for laprompt mode [variance, covariance, multi-centroid]')
    parser.add_argument('--laprompt_ema_decay', type=float, default=0.0, help='ema decay factor for laprompt prompt attention update')
    parser.add_argument('--laprompt_tuned_epoch', type=int, default=1, help='task-level tuning epochs for laprompt mode')
    parser.add_argument('--laprompt_n_centroids', type=int, default=5, help='number of centroids per class for laprompt multi-centroid storage')
    parser.add_argument('--laprompt_add_num', type=int, default=8, help='minimum synthetic samples per class component during classifier alignment')
    parser.add_argument('--laprompt_backbone_name', type=str, default='vit_base_patch16_224', help='timm backbone name for laprompt model')
    parser.add_argument('--laprompt_pretrained', action=argparse.BooleanOptionalAction, default=True, help='enable pretrained timm weights for laprompt model')
    parser.add_argument('--laprompt_use_task_token', action='store_true', help='enable task embedding in laprompt model')
    parser.add_argument('--laprompt_max_tasks', type=int, default=100, help='maximum task ids for laprompt task embedding')
    parser.add_argument('--pool_size', type=int, default=10, help='laprompt prompt pool size')
    parser.add_argument('--length', type=int, default=5, help='laprompt prompt length')
    parser.add_argument('--top_k', type=int, default=1, help='laprompt top-k prompts')
    parser.add_argument('--prompt_layer_idx', type=int, nargs='*', default=None, help='laprompt prompt injection layers, e.g. --prompt_layer_idx 0 1 2')
    parser.add_argument('--temperature', type=float, default=1.0, help='laprompt prompt routing temperature')
    parser.add_argument('--laprompt_use_self_attn', action='store_true', help='enable prompt self-attention in laprompt pool')
    parser.add_argument('--laprompt_batchwise_prompt', action=argparse.BooleanOptionalAction, default=True, help='enable batchwise prompt selection in laprompt pool')
    parser.add_argument('--laprompt_use_layer_embedding', action=argparse.BooleanOptionalAction, default=True, help='enable layer embedding in laprompt prompt router')

    # LaPrompt compatibility aliases (for laprompt_mix-like scripts)
    parser.add_argument('--tuned_epoch', type=int, help='alias of --laprompt_tuned_epoch')
    parser.add_argument('--ca_lr', type=float, help='alias of --laprompt_ca_lr')
    parser.add_argument('--crct_epochs', type=int, help='alias of --laprompt_ca_epochs')
    parser.add_argument('--ca_storage_efficient_method', type=str, help='alias of --laprompt_ca_storage')
    parser.add_argument('--n_centroids', type=int, help='alias of --laprompt_n_centroids')
    parser.add_argument('--add_num', type=int, help='alias of --laprompt_add_num')
    parser.add_argument('--ema_decay', type=float, help='alias of --laprompt_ema_decay')
    parser.add_argument('--freeze', action=argparse.BooleanOptionalAction, default=True, help='freeze ViT backbone for laprompt prefix-tuning')
    
    # LaPrompt Online Learning Optimization
    parser.add_argument('--use_dynamic_logit_mask', action=argparse.BooleanOptionalAction, default=True, help='enable dynamic logit masking for online learning')
    parser.add_argument('--use_forgetting_aware_init', action=argparse.BooleanOptionalAction, default=True, help='enable forgetting-aware prompt initialization')
    parser.add_argument('--logit_mask_temp', type=float, default=1.0, help='temperature for logit mask scaling')
    parser.add_argument('--init_strategy', type=str, default='variance_perturb', help='forgetting-aware init strategy: variance_perturb, knn_prototype, gaussian_sample, task_orthogonal, mean_centered, contrastive_init, zero_init, random_scaled')
    parser.add_argument('--init_scale', type=float, default=0.01, help='scale factor for forgetting-aware initialization')

    # parser.add_argument('--beta', type=float, default=0., help='# candidates to use for peeking into the updated head')
    # parser.add_argument('--charlie', type=float, default=0., help='# candidates to use for CP hyperparameter')
    

    # Ours
    # parser.add_argument('--use_mask', action='store_true', help='use mask for our method')
    # parser.add_argument('--use_contrastiv', action='store_true', help='use contrastive loss for our method')
    # parser.add_argument('--use_last_layer', action='store_true', help='use last layer for our method')
    # parser.add_argument('--use_dyna_exp', action='store_true', help='use dynamic expand for our method')

    args = parser.parse_args()
    return args

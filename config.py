import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rng_seed', type=int, default=0)

    # model and task
    parser.add_argument('--task_name', type=str, default='nli', choices=['nli', 'hsd'])
    parser.add_argument('--train_dataset_name', type=str, default='mnli')
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-small', choices=[
        'roberta-large', 'bert-large-uncased', 'prajjwal1/bert-tiny', 
        'microsoft/deberta-v3-small', 'microsoft/deberta-v3-base', 'microsoft/deberta-v3-large', 
        'google/electra-small-discriminator', 'google/electra-base-discriminator', 'google/electra-large-discriminator',
        ])
    
    # training/evaluation specification
    parser.add_argument('--mode', type=str, default='erm', choices=['erm', 'dm'])
    parser.add_argument('--erm_loss_func', type=str, default='cross_entropy', choices=['cross_entropy'])
    parser.add_argument('--max_train_steps', type=int, default=None, help='number of training steps.')
    parser.add_argument('--max_train_epochs', type=int, default=3,
                        help='number of training epochs (will be ignored if ``max_train_steps`` is provided).')
    parser.add_argument('--eval_interval', type=int, default=None)

    # optimization
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])  
    parser.add_argument('--scheduler', type=str, default='linear_with_warmup', choices=[
        'linear_with_warmup', 'reduce_lr_on_plateau', 'constant', 'constant_with_warmup', 'cosine_with_warmup',
    ])
    # parser.add_argument('--sgd_scheduler', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--mixed_precision', type=str, default='bf16', choices=['no', 'bf16', 'fp16'])
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=2e-05)  
    parser.add_argument('--warmup_rate', type=float, default=0.1)  
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=0)  
    parser.add_argument('--no_grad_clip', action='store_true')
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--use_gradient_checkpointing', action='store_true')

    # reference specification
    parser.add_argument('--reference_run_dir', type=str, default=None)
    
    parser.add_argument('--dm_filter_rate', default=0.33, type=float)
    parser.add_argument('--dm_filter_type', default='ambiguous', type=str, choices=[
        'ambiguous', 'hard_to_learn', 'easy_to_learn', 'random'])

    # save and load
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_every', action='store_true')
    parser.add_argument('--save_last', action='store_true')
    parser.add_argument('--save_at_epochs', nargs='*', default=[])
    parser.add_argument('--save_at_steps', nargs='*', default=[])
    parser.add_argument('--no_checkpoint', action='store_true')
    # wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project_name', type=str, default='ftft')
    parser.add_argument('--wandb_run_name', type=str, default=None)

    args = parser.parse_args()
    return args
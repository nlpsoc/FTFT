import math
import os
import warnings

os.environ['TRANSFORMERS_CACHE'] = 'data/hg_data/transformers'
os.environ['HF_DATASETS_CACHE'] = 'data/hg_data/datasets'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoConfig, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding, 
)
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, broadcast

import utils
from dataloader import DATALOADER_DICT
import trainers
from config import parse_args


def main():
    args = parse_args()
    set_seed(args.rng_seed)
    # Set up save directory and save config
    args.save_dir = utils.generate_save_name(args) if not args.save_dir else args.save_dir

    # Initialize accelerator
    accelerator_kwargs = {
        'mixed_precision': args.mixed_precision, 
        'gradient_accumulation_steps': args.gradient_accumulation_steps
    }
    if args.use_wandb:
        accelerator_kwargs['log_with'] = 'wandb'
    accelerator = Accelerator(**accelerator_kwargs)
    if args.use_wandb:
        wandb_kwargs = {'wandb': {'name': args.wandb_run_name}} if args.wandb_run_name is not None else {}
        accelerator.init_trackers(args.wandb_project_name, config=utils.wandb_process_args(args), init_kwargs=wandb_kwargs)

    # Set up logging
    if not accelerator.is_local_main_process:
        warnings.filterwarnings('ignore')
    if accelerator.is_local_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        _ = utils.write_json(vars(args), os.path.join(args.save_dir, 'config.json'))

    accelerator.wait_for_everyone()
    logger = get_logger(__name__, log_level='INFO')
    log_filename = os.path.join(args.save_dir, 'train.log')
    logger = utils.setup_logger(logger, log_filename)
    logger.info(f'Model directory: {args.save_dir}')

    # Load data
    logger.info('Loading data')
    dataloader = DATALOADER_DICT[args.task_name](
        max_length=args.max_length, train_dataset_name=args.train_dataset_name)
    with accelerator.main_process_first():  # Re-use generated data cache of the first process
        train_data, train_info = dataloader.load_train(model_name=args.model_name)
        # copy uid column from train_info to train_data
        train_data = train_data.add_column('uid', train_info['uid'])
        eval_data_info_dict = dataloader.load_dev(model_name=args.model_name)
        test_data_info_dict = dataloader.load_test(model_name=args.model_name)
    eval_data_dict = {dataset_name: eval_data_info_dict[dataset_name][0] for dataset_name in eval_data_info_dict}
    test_data_dict = {dataset_name: test_data_info_dict[dataset_name][0] for dataset_name in test_data_info_dict}

    # Initialize data collator 
    data_collator = DataCollatorWithPadding(dataloader.tokenizer_dict[args.model_name])
    # Write extra info to args
    args.num_labels, args.num_samples = \
        len(dataloader.dataset_info[dataloader.train_dataset_name]['label_names']), max(train_data['data_idx']) + 1
    # Change batch size accordingly 
    assert args.train_batch_size % int(
        accelerator.num_processes * args.gradient_accumulation_steps) == 0 and \
        args.eval_batch_size % accelerator.num_processes == 0 
    args.train_batch_size = int(
        args.train_batch_size // int(accelerator.num_processes * args.gradient_accumulation_steps))
    args.eval_batch_size = int(args.eval_batch_size // accelerator.num_processes)
    accelerator.wait_for_everyone()

    # Load model and optimizer
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # Call training functions
    if args.mode == 'erm':
        args.max_train_steps = args.max_train_steps if args.max_train_steps is not None \
            else args.max_train_epochs * math.ceil(len(train_data) / (
            args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps))
        logger.info(f'Total number of training steps: {args.max_train_steps}')
        optimizer, scheduler = utils.construct_optimizer_scheduler(model, args)
        trainers.train(model, train_data, eval_data_dict, test_data_dict, 
                            optimizer, scheduler, data_collator, accelerator, args)

    elif args.mode == 'dm': 
        # Construct data map indices only in the main process and broadcast to other processes to ensure consistency
        if accelerator.is_local_main_process:
            if args.dm_filter_type == 'random':
                permuted_indices = torch.randperm(len(train_data))
                train_data_uid = train_data['uid']
                selected_uid_list = [train_data_uid[idx] for idx in permuted_indices[
                    :int(len(train_data) * args.dm_filter_rate)]]

            elif args.dm_filter_type in ['ambiguous', 'hard_to_learn']:
                assert args.reference_run_dir is not None
                uid_to_logits = torch.load(os.path.join(args.reference_run_dir, 'uid_to_logits.pt'), map_location='cpu')
                selected_uid_list = utils.obtain_data_map_uid_list(
                    train_data, uid_to_logits, args.dm_filter_type, args.dm_filter_rate)

            select_uid_set = set(selected_uid_list)

        # Filter train data
        select_uid_set = broadcast(select_uid_set, from_process=0)
        filtered_train_data = train_data.filter(lambda example: example['uid'] in select_uid_set)
        filtered_train_data = filtered_train_data.map(lambda _, idx: {'data_idx': idx}, with_indices=True)

        args.num_samples = max(filtered_train_data['data_idx']) + 1
        args.max_train_steps = args.max_train_steps if args.max_train_steps is not None \
            else args.max_train_epochs * math.ceil(len(filtered_train_data) / (
            args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps))
        logger.info(f'Total number of training steps: {args.max_train_steps}')
        optimizer, scheduler = utils.construct_optimizer_scheduler(model, args)
        trainers.train(model, filtered_train_data, eval_data_dict, test_data_dict, 
                        optimizer, scheduler, data_collator, accelerator, args)
    
    if args.use_wandb:
        accelerator.end_training()


if __name__ == '__main__':
    main()

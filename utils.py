import os
import json
import re
import logging
import math

import torch
from torch import nn as nn
from torch.optim import AdamW, SGD
from tqdm.auto import tqdm
from transformers import get_scheduler
import evaluation as evaluate


# Forward
def forward_model(model, batch, output_hidden_states=False):
    useful_keys = ['input_ids', 'attention_mask', 'token_type_ids']
    input_batch = {key : batch[key] for key in set(useful_keys) & set(batch.keys())}
    outputs = model(**input_batch, output_hidden_states=output_hidden_states)
    if output_hidden_states:
        return outputs.logits, outputs.hidden_states
    else:
        return outputs.logits


# Optimization
def get_optimizer_grouped_parameters(model, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,  # Weight decay group
        },
        {
            "params": [p for n, p in model.named_parameters() 
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,  # No weight decay group
        },
    ]
    return optimizer_grouped_parameters


def construct_optimizer(models, args):
    # Get parameter groups: some parameters may not need weight decay
    if not isinstance(models, list):
        models = [models]
    optimizer_grouped_parameters = []
    for model in models:
        optimizer_grouped_parameters += get_optimizer_grouped_parameters(model, args.weight_decay)
    
    optimizer = None
    if args.optimizer == 'adam':
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_eps)
    elif args.optimizer == 'sgd':
        optimizer = SGD(optimizer_grouped_parameters, lr=args.learning_rate, momentum=0.9)

    return optimizer


def construct_scheduler(optimizer, args):
    num_warmup_steps = int(args.warmup_rate * args.max_train_steps) if args.warmup_rate < 1 else int(args.warmup_rate)

    if args.scheduler == 'linear_with_warmup':
        scheduler_name = 'linear'
    elif args.scheduler == 'cosine_with_warmup':
        scheduler_name = 'cosine'
    else:
        scheduler_name = args.scheduler

    return get_scheduler(
        name=scheduler_name, optimizer=optimizer, 
        num_warmup_steps=num_warmup_steps, num_training_steps=int(args.max_train_steps))


def construct_optimizer_scheduler(models, args):
    optimizer = construct_optimizer(models, args)
    scheduler = construct_scheduler(optimizer, args)
    return optimizer, scheduler


# IO
def setup_logger(logger, log_filename=None):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not logger.logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.logger.addHandler(console_handler)
    if log_filename is not None:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        logger.logger.addHandler(file_handler)
    return logger


def save_model(save_path, accelerator, prefix='validation', model=None, eval_result=None, eval_logits_dict=None):
    if model is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path, 
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
    if eval_result is not None and accelerator.is_local_main_process:
        write_json(eval_result, os.path.join(save_path, f'{prefix}_result.json'))
    if eval_logits_dict is not None and accelerator.is_local_main_process:
        for dataset_name in eval_logits_dict:
            torch.save(eval_logits_dict[dataset_name], os.path.join(save_path, f'{prefix}_{dataset_name}_logits.pt'))


def write_json(data, filename, formatted=True):
    with open(filename, 'w') as fout:
        if formatted:
            json.dump(data, fout, indent=4, separators=(',', ':'))
        else:
            json.dump(data, fout)
    return filename


def load_json(file_path):
    with open(file_path) as fin:
        return json.load(fin)


def data_indices_dict_to_uid(data_indices_dict, uid_list):
    data_uid_dict = {}
    for epoch in data_indices_dict:
        data_indices = data_indices_dict[epoch].cpu().numpy().tolist()
        data_uid_dict[epoch] = [uid_list[idx] for idx in data_indices]
    return data_uid_dict


def logits_dict_to_uid(logits_dict, uid_list):
    logits = torch.stack(list(logits_dict.values()), dim=0)
    # if any row is [0, 0, 0], then discard this epoch
    if (logits[-1] == 0).all(dim=-1).any():
        logits = logits[:-1]
    return {uid: logits[:, idx, :] for idx, uid in enumerate(uid_list)}


def generate_save_name(args):
    model_name = args.model_name
    if '/' in model_name:
        model_name = model_name.replace('/', '-')
    save_dir = f'ckpt/{args.task_name}/{args.mode}/{model_name}'
    if not os.path.exists(save_dir):
        max_run_idx = -1
    else:
        max_run_idx = 0
        for model_dir in os.listdir(save_dir):
            match = re.match(r'run_(\d+)', model_dir)
            if match:
                run_idx = int(match.group(1))
                max_run_idx = run_idx if run_idx > max_run_idx else max_run_idx
    return os.path.join(save_dir, f'run_{max_run_idx + 1}')


def wandb_process_args(args):
    main_model_to_name_size = {
        'roberta-large': ('RoBERTa', 'Large'), 
        'bert-large-uncased': ('BERT', 'Large'),
        'prajjwal1/bert-tiny': ('BERT', 'Tiny'),
        'microsoft/deberta-v3-small': ('DeBERTaV3', 'Small'),
        'microsoft/deberta-v3-base': ('DeBERTaV3', 'Base'),
        'microsoft/deberta-v3-large': ('DeBERTaV3', 'Large'),
        'google/electra-small-discriminator': ('ELECTRA', 'Small'),
        'google/electra-base-discriminator': ('ELECTRA', 'Base'),
        'google/electra-large-discriminator': ('ELECTRA', 'Large'),
    }
    ref_model_name_dict = {'roberta': 'RoBERTa', 'bert': 'BERT', 'debertav3': 'DeBERTaV3', 'electra': 'ELECTRA'}

    wandb_config = vars(args)

    reference_run_dir = wandb_config['reference_run_dir']
    if reference_run_dir is None:
        wandb_config['ref_model'] = None
        wandb_config['ref_size'] = None
    else:
        reference_run_dir = reference_run_dir.split('/')
        ref_dataset_name, ref_model_info, split_seed_info = \
            reference_run_dir[1], reference_run_dir[3], reference_run_dir[4].split('_')
        wandb_config['ref_dataset_name'] = ref_dataset_name
        ref_model, ref_size = ref_model_info.split('_')
        wandb_config['ref_model'], wandb_config['ref_size'] = ref_model_name_dict[ref_model], ref_size.capitalize()
        wandb_config['ref_split'], wandb_config['ref_seed'] = int(split_seed_info[1]), int(split_seed_info[3])

    wandb_config['main_model'], wandb_config['main_size'] = main_model_to_name_size[wandb_config['model_name']]
    return wandb_config


def compute_save_at_steps(num_samples, num_processes, eval_interval, args):
    save_at_steps = []
    num_steps_in_one_epoch =  math.ceil(num_samples / (
        args.train_batch_size * num_processes * args.gradient_accumulation_steps))

    # Priority: 1) save_at_steps, 2) save_at_epochs > 0, 3) save_every and save_last
    if len(args.save_at_steps) > 0:
        save_at_steps = [int(step) for step in args.save_at_steps]
    elif len(args.save_at_epochs) > 0:  # epoch number starts from 0
        save_at_steps = [num_steps_in_one_epoch * (int(epoch) + 1) for epoch in args.save_at_epochs]
    elif args.save_every or args.save_last:
        if args.save_every:
            save_at_steps = list(range(eval_interval, args.max_train_steps, eval_interval))
        save_at_steps.append(args.max_train_steps)

    return set([step for step in save_at_steps if step % eval_interval == 0 and step <= args.max_train_steps])


# Evaluation
def eval_acc(model, dataloader, accelerator, return_logits=False, metric_name='acc'):
             
    model.eval()
    if metric_name == 'acc':
        metric = evaluate.load('accuracy')
    elif metric_name == 'macro_f1' or metric_name == 'micro_f1':
        metric = evaluate.load('f1')
    else:
        raise ValueError(f'Unsupported metric {metric_name}')
    with torch.no_grad():
        dataset_logits, dataset_data_indices = [], []
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            logits = forward_model(model, batch)
            # merge logits if needed (e.g., merge contradiction and neutral to non-entailment for HANS)
            predictions = logits.argmax(dim=-1)
            batch_labels = batch['restricted_labels'] if 'restricted_labels' in batch else batch['labels']
            predictions, references = accelerator.gather_for_metrics((predictions, batch_labels))
            metric.add_batch(predictions=predictions.cpu(), references=references.cpu())
            if return_logits:
                dataset_logits.append(accelerator.gather(logits))
                dataset_data_indices.append(accelerator.gather(batch['data_idx']))
    accelerator.wait_for_everyone()
    if metric_name == 'acc':
        performance = metric.compute()['accuracy']
    elif metric_name == 'macro_f1':
        performance = metric.compute(average='macro')['f1']
    elif metric_name == 'micro_f1':
        performance = metric.compute(average='micro')['f1']
    
    # concatenate logits by the order of data_idx
    if return_logits:
        dataset_logits, dataset_data_indices = \
            torch.cat(dataset_logits, dim=0).type(torch.float32).cpu(), \
            torch.cat(dataset_data_indices, dim=0).cpu()
        dataset_logits_clone = torch.zeros_like(dataset_logits)
        dataset_logits_clone[dataset_data_indices] = dataset_logits

    return performance, dataset_logits_clone if return_logits else None


# DM functions
def compute_data_map_scores(uid_list, uid_to_logits, uid_to_label):
    probs = torch.stack([uid_to_logits[uid] for uid in uid_list], dim=0).softmax(dim=-1)  # (N, E, C)
    labels = torch.LongTensor([uid_to_label[uid] for uid in uid_list])  # (N,)
    conf = probs[torch.arange(len(uid_list)), :, labels]  # (N, E)
    return conf.std(dim=1), -conf.mean(dim=1)


def obtain_data_map_uid_list(train_data, uid_to_logits, mode='ambiguous', filter_rate=0.33):
    uid_list = list(uid_to_logits.keys())
    uid_to_label = {uid: label for uid, label in zip(train_data['uid'], train_data['labels'])}
    ambiguous_scores, hard_scores = compute_data_map_scores(uid_list, uid_to_logits, uid_to_label)
    # which score to use
    if mode == 'ambiguous':
        scores = ambiguous_scores
    elif mode == 'hard_to_learn':
        scores = hard_scores

    top_k_uid_indices = torch.arange(len(uid_list))[scores > torch.quantile(scores, 1 - filter_rate)]
    return [uid_list[idx] for idx in top_k_uid_indices.numpy().tolist()]

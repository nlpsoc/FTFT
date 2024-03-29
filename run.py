# Description: This file constructs the command line scripts for the project.
import os
import argparse
import yaml


def read_yaml(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_hf_name(model, size):
    if model == "debertav3":
        return f"microsoft/deberta-v3-{size}"
    elif model == "electra":
        return f"google/electra-{size}-discriminator"
    elif model == "roberta":
        return f"roberta-{size}"
    elif model == "bert":
        if size == "tiny":
            return "prajjwal1/bert-tiny"
        else:
            return f"bert-{size}-uncased"
    else:
        raise ValueError(f"Unsupported model: {model}")


def construct_erm_commands(config, rng_seed):
    script_command = f"accelerate launch --config_file {config['environment']['accelerate_config']} "\
        f"train.py --rng_seed {rng_seed}"
    
    data_args = {**config['data']['main']}
    data_command = ' '.join([f"--{k} {v}" for k, v in data_args.items() if v is not None])

    hf_name = get_hf_name(config['model']['main']['model'], config['model']['main']['size'])
    model_command = f"--model_name {hf_name}"

    optim_args = {'mode': 'erm', **config['optim']['main']}
    optim_command = ' '.join([f"--{k} {v}" for k, v in optim_args.items() if v is not None])

    save_dir = os.path.join(
        "ckpt", config['data']['main']['train_dataset_name'], 
        config['log']['main']['project_name'], 
        f"{config['model']['main']['model']}_{config['model']['main']['size']}", 
        f"seed_{rng_seed}_steps_{config['optim']['main']['max_train_steps']}"
    )
    wandb_run_name = f"data_{config['data']['main']['train_dataset_name']}_"\
        f"model_{config['model']['main']['model']}_{config['model']['main']['size']}_"\
        f"seed_{rng_seed}_steps_{config['optim']['main']['max_train_steps']}"
    log_args = {'save_dir': save_dir, **{k: v for k, v in config['log']['main'].items() if k != 'project_name'}}
    log_command = ' '.join([f"--{k} {v}" for k, v in log_args.items() if k != 'use_wandb' and v is not None])
    if log_args['use_wandb']:
        log_command += " --use_wandb"
        log_command += f' --wandb_project_name {config["log"]["main"]["project_name"]}'
        log_command += f" --wandb_run_name {wandb_run_name}"


    command = f"{script_command} {data_command} {model_command} {optim_command} {log_command}"

    return command


def construct_dm_commands(config, rng_seed):
    script_command = f"accelerate launch --config_file {config['environment']['accelerate_config']} "\
        f"train.py --rng_seed {rng_seed}"

    data_args = {**config['data']['main']}
    data_command = ' '.join([f"--{k} {v}" for k, v in data_args.items() if v is not None])

    hf_name = get_hf_name(config['model']['main']['model'], config['model']['main']['size'])
    model_command = f"--model_name {hf_name}"

    optim_args = {'mode': 'dm', **config['optim']['main']}
    optim_command = ' '.join([f"--{k} {v}" for k, v in optim_args.items() if v is not None])
    if config['optim']['main']['dm_filter_type'] != 'random':
        ref_dir = os.path.join(
            "ckpt", config['data']['main']['train_dataset_name'], 
            config['log']['ref']['project_name'], 
            f"{config['model']['ref']['model']}_{config['model']['ref']['size']}", 
            f"seed_{rng_seed}_steps_{config['optim']['ref']['max_train_steps']}"
        )
        optim_command += f" --reference_run_dir {ref_dir}"
        

    if config['optim']['main']['dm_filter_type'] == 'random':
        save_dir = os.path.join(
            "ckpt", config['data']['main']['train_dataset_name'], 
            config['log']['main']['project_name'], 
            f"{config['model']['main']['model']}_{config['model']['main']['size']}", 
            f"type_{config['optim']['main']['dm_filter_type']}_"\
            f"seed_{rng_seed}_steps_{config['optim']['main']['max_train_steps']}"
        )
        wandb_run_name = f"data_{config['data']['main']['train_dataset_name']}_"\
            f"model_{config['model']['main']['model']}_{config['model']['main']['size']}_"\
            f"type_{config['optim']['main']['dm_filter_type']}_"\
            f"seed_{rng_seed}_steps_{config['optim']['main']['max_train_steps']}"
    else:
        save_dir = os.path.join(
            "ckpt", config['data']['main']['train_dataset_name'], 
            config['log']['main']['project_name'], 
            f"{config['model']['main']['model']}_{config['model']['main']['size']}", 
            f"ref_{config['model']['ref']['model']}_{config['model']['ref']['size']}_"\
            f"steps_{config['optim']['ref']['max_train_steps']}_"\
            f"type_{config['optim']['main']['dm_filter_type']}_"\
            f"seed_{rng_seed}_steps_{config['optim']['main']['max_train_steps']}"
        )
        wandb_run_name = f"data_{config['data']['main']['train_dataset_name']}_"\
            f"model_{config['model']['main']['model']}_{config['model']['main']['size']}_"\
            f"ref_{config['model']['ref']['model']}_{config['model']['ref']['size']}_"\
            f"steps_{config['optim']['ref']['max_train_steps']}_"\
            f"type_{config['optim']['main']['dm_filter_type']}_"\
            f"seed_{rng_seed}_steps_{config['optim']['main']['max_train_steps']}"

    log_args = {'save_dir': save_dir, **{k: v for k, v in config['log']['main'].items() if k != 'project_name'}}
    log_command = ' '.join([f"--{k} {v}" for k, v in log_args.items() if k != 'use_wandb' and v is not None])
    if log_args['use_wandb']:
        log_command += " --use_wandb"
        log_command += f' --wandb_project_name {config["log"]["main"]["project_name"]}'
        log_command += f" --wandb_run_name {wandb_run_name}"

    log_command += ' --save_last'

    command = f"{script_command} {data_command} {model_command} {optim_command} {log_command}"
    return command


def construct_train_commands(config):
    if config["mode"] == "dm":
        train_command_func = construct_dm_commands
    elif config["mode"] == "erm":
        train_command_func = construct_erm_commands
    else:
        raise ValueError(f"Unsupported mode: {config['mode']}")
    
    train_commands = ['#!/bin/bash -x\n']
    for rng_seed in config["rng_seeds"]:
        train_command = train_command_func(config, rng_seed)
        train_commands.append(train_command)
    return "\n".join(train_commands)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    args = parser.parse_args()

    config_path = args.config_path
    # read the YAML file
    config = read_yaml(config_path)
    # Control training duration by setting `max_train_steps`
    assert config['optim']['main']['max_train_steps'] is not None
    
    commands = construct_train_commands(config)
    output_path = config_path.replace("yaml", "sh")
    with open(output_path, "w") as file:
        file.write(commands)
    print(f"Output bash file: {output_path}")


if __name__ == "__main__":
    main()

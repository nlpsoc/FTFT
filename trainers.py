import os
import math

import torch
import torch.utils.data as torch_data
from tqdm.auto import tqdm
from accelerate.logging import get_logger

import utils


logger = get_logger(__name__, log_level='INFO')

EVAL_METRIC_DICT = {
    'nli': 'acc', 
    'hsd': 'macro_f1', 
}


def evaluate_model(model, eval_dataloader_dict, accelerator, metric_name='acc'): 
    model.eval()
    evaluation_result, evaluation_logits_dict = {}, {}
    for dataset_name in eval_dataloader_dict:
        accelerator.print(f'Evaluating on {dataset_name}')
        dataset_performance, dataset_logits = utils.eval_acc(
            model, eval_dataloader_dict[dataset_name], accelerator, return_logits=True, metric_name=metric_name)
        evaluation_result[dataset_name] = dataset_performance
        evaluation_logits_dict[dataset_name] = dataset_logits
    evaluation_str = '\n'.join([f'{dataset_name} {metric_name}: ' + '{:.4f}'.format(evaluation_result[dataset_name])
                                for dataset_name in evaluation_result])
    accelerator.print(evaluation_str)

    model.train()
    return evaluation_result, evaluation_logits_dict


def compute_ce_loss(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels, reduction='none')


def compute_loss(logits, batch, mode='erm'):
    if mode in ['erm', 'dm']:
        loss = compute_ce_loss(logits, batch['labels']).mean()
    return loss


def train(model, train_data, eval_data_dict, test_data_dict, 
          optimizer, scheduler, data_collator, accelerator, args, 
          method_args=None):

    model.train()
    # Save and remove uid from train_data
    uid_list = train_data['uid']
    train_data = train_data.remove_columns(['uid'])

    # Prepare dataloaders
    train_dataloader = torch_data.dataloader.DataLoader(
        train_data, shuffle=True, batch_size=args.train_batch_size, collate_fn=data_collator)
    eval_dataloader_dict = {dataset_name: torch_data.dataloader.DataLoader(
        eval_data_dict[dataset_name], shuffle=False, batch_size=args.eval_batch_size, collate_fn=data_collator)
        for dataset_name in eval_data_dict}
    test_dataloader_dict = {dataset_name: torch_data.dataloader.DataLoader(
        test_data_dict[dataset_name], shuffle=False, batch_size=args.eval_batch_size, collate_fn=data_collator)
        for dataset_name in test_data_dict}
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader)
    eval_dataloader_dict = {dataset_name: accelerator.prepare(eval_dataloader_dict[dataset_name]) 
                            for dataset_name in eval_data_dict}
    test_dataloader_dict = {dataset_name: accelerator.prepare(test_dataloader_dict[dataset_name]) 
                            for dataset_name in test_dataloader_dict}

    # Evaluation interval should be specified in terms of mini-steps per update
    eval_interval = args.eval_interval if args.eval_interval else \
        math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    save_at_steps = utils.compute_save_at_steps(len(train_data), accelerator.num_processes, eval_interval, args)
    accelerator.wait_for_everyone()

    completed_steps, avg_loss, epoch_idx = 0, 0, 0
    # Save logits and data indices for each epoch
    logits_dict, data_indices_dict = {}, {} 
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    while completed_steps < args.max_train_steps:
        logits_dict[epoch_idx] = torch.zeros((args.num_samples, args.num_labels))
        epoch_logits, epoch_data_indices = [], []
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                logits = utils.forward_model(model, batch)
                loss = compute_loss(logits, batch, mode=args.mode)

                accelerator.backward(loss)
                if not args.no_grad_clip: 
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.scheduler != 'reduce_lr_on_plateau':
                    if not accelerator.optimizer_step_was_skipped:
                        scheduler.step()  
                optimizer.zero_grad()
            
            # Average loss and steps should be computed each gradient accumulation step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                avg_loss = (completed_steps - 1) / completed_steps * avg_loss + loss.detach() / completed_steps
                if args.use_wandb:
                    accelerator.log({
                        'step': completed_steps,
                        'train_loss': loss.detach().item(), 
                        'train_avg_loss': avg_loss.item()
                    })

            data_indices, logits = accelerator.gather((batch['data_idx'], logits))
            epoch_data_indices.append(data_indices.cpu())
            epoch_logits.append(logits.detach().type(torch.float32).cpu())

            # evaluation
            if (completed_steps % eval_interval == 0 or completed_steps == args.max_train_steps) \
                and completed_steps != 0 and not args.no_checkpoint and accelerator.sync_gradients:
                model.eval()
                avg_loss = accelerator.reduce(avg_loss, reduction='mean')
                accelerator.print(f'Evaluating at step #{completed_steps} in epoch {epoch_idx}')
                accelerator.print(f'average loss: {avg_loss.item()}')
                # Run reduce on plateau scheduler
                if args.scheduler == 'reduce_lr_on_plateau':
                    scheduler.step(avg_loss)

                validation_result, validation_logits = evaluate_model(
                    model, eval_dataloader_dict, accelerator, metric_name=EVAL_METRIC_DICT[args.task_name])
                test_result, test_logits = evaluate_model(
                    model, test_dataloader_dict, accelerator, metric_name=EVAL_METRIC_DICT[args.task_name])
                if args.use_wandb and accelerator.is_local_main_process:
                    accelerator.log({
                        'step': completed_steps,  
                        **{f'val_{dataset_name}': validation_result[dataset_name] for dataset_name in validation_result}, 
                        **{f'test_{dataset_name}': test_result[dataset_name] for dataset_name in test_result}, 
                    })

                accelerator.print(f'Saving at step #{completed_steps} in epoch {epoch_idx}')
                save_path = os.path.join(args.save_dir, f'step_{completed_steps}')
                if accelerator.is_local_main_process:
                    os.makedirs(save_path, exist_ok=True)
                accelerator.wait_for_everyone()
                utils.save_model(
                    save_path, accelerator, prefix='validation', 
                    model=model if completed_steps in save_at_steps else None, 
                    eval_result=validation_result, eval_logits_dict=validation_logits)
                utils.save_model(
                    save_path, accelerator, prefix='test', model=None, 
                    eval_result=test_result, eval_logits_dict=test_logits)

            model.train()
            if completed_steps == args.max_train_steps:
                break

        # End of epoch: save logits and data indices
        epoch_logits, epoch_data_indices = torch.cat(epoch_logits, dim=0), torch.cat(epoch_data_indices, dim=0)
        data_indices_dict[epoch_idx] = epoch_data_indices
        logits_dict[epoch_idx][data_indices_dict[epoch_idx]] = epoch_logits
        epoch_idx += 1

    model.eval()
    # save the final model if necessary
    save_path = os.path.join(args.save_dir, f'step_{completed_steps}')
    if not os.path.exists(save_path):
        accelerator.print(f'Evaluating and saving the final model at step #{completed_steps} in epoch {epoch_idx}')
        if accelerator.is_local_main_process:
            os.makedirs(save_path, exist_ok=True)
        validation_result, validation_logits = evaluate_model(
            model, eval_dataloader_dict, accelerator, metric_name=EVAL_METRIC_DICT[args.task_name])
        test_result, test_logits = evaluate_model(
            model, test_dataloader_dict, accelerator, metric_name=EVAL_METRIC_DICT[args.task_name])
        accelerator.wait_for_everyone()
        utils.save_model(
            save_path, accelerator, prefix='validation', 
            model=model if completed_steps in save_at_steps else None, 
            eval_result=validation_result, eval_logits_dict=validation_logits)
        utils.save_model(
            save_path, accelerator, prefix='test', model=None, 
            eval_result=test_result, eval_logits_dict=test_logits)

    if accelerator.is_local_main_process:
        uid_to_logits = utils.logits_dict_to_uid(logits_dict, uid_list)
        data_uid_dict = utils.data_indices_dict_to_uid(data_indices_dict, uid_list)
        torch.save(uid_to_logits, os.path.join(args.save_dir, 'uid_to_logits.pt'))
        torch.save(data_uid_dict, os.path.join(args.save_dir, 'data_uid_dict.pt'))

    return model, logits_dict

    
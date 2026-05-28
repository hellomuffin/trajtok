import pandas as pd
import time
import datetime
import os
import wandb
from os.path import join
import logging
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from trajtok_segmenter.data import create_dataset, create_sampler, create_loader, MetaLoader
from trajtok_segmenter.train.retrieval_utils import evaluation_wrapper
from trajtok_segmenter.train.shared_utils import setup_model, identify_model_cls
from trajtok_segmenter.train.config_utils import setup_main
from trajtok_segmenter.train.basic_utils import MetricLogger, SmoothedValue, setup_seed, remove_files_if_exist
from trajtok_segmenter.train.distributed import get_rank, get_world_size, is_main_process, init_distributed_mode
from trajtok_segmenter.train.logger import log_dict_to_wandb, setup_wandb


logger = logging.getLogger(__name__)



def _save_step_checkpoint(model_without_ddp, optimizer, scheduler, scaler, config, epoch, global_step):
    """Atomic step-based checkpoint save (rank-0 only). Writes to a tmp path
    then renames so a partial file from a crash never overwrites a good one.
    Epoch-end checkpoints are still written separately by main()."""
    if not is_main_process():
        return
    save_obj = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
    }
    final_path = join(config.output_dir, "latest.pth")
    tmp_path = final_path + ".tmp"
    torch.save(save_obj, tmp_path)
    os.replace(tmp_path, final_path)
    logger.info(f"Saved step checkpoint at step {global_step} -> {final_path}")


def train(model, train_loaders, optimizer, tokenizer, epoch, global_step, device, scheduler, scaler, config, train_segmenter=False, prefix='train/'):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=30, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=30, fmt="{value:.4f}"))

    log_names = ["loss_ita"]
    if train_segmenter: log_names += ["loss_low_res_pixel", "loss_low_res_class", "loss_high_res_pixel", "loss_high_res_class"]
    
    media_types = [loader.dataset.media_type for loader in train_loaders]
    for name in log_names:
        for m in media_types:
            metric_logger.add_meter(f"{m}-{name}", SmoothedValue(window=30, fmt="{value:.4f}"))

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            if hasattr(d.sampler, 'set_epoch'):
                d.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    
    for i, batch in enumerate(iterator):
        
        media_type, real_batch = batch
        
        image, text, idx, segment, graph, num_tokens = real_batch

        image = image.to(device, non_blocking=True)
        
        if config.vit_type == 'trajvit':
            segment = segment.to(device, non_blocking=True)
            graph = graph.to(device, non_blocking=True)
            num_tokens = num_tokens.to(device, non_blocking=True)
        
        text_input = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=config.max_txt_l[media_type], return_tensors="pt"
        ).to(device)  # change from "longest" to "max_length"

        with torch.cuda.amp.autocast(enabled=config.fp16):
            loss_dict = model((image, segment, graph, num_tokens), text_input, idx=None)
            loss = sum([loss_dict[k] for k in log_names])
            
        # check if any rank produces NaN loss 
        flag = torch.tensor(
            [0 if torch.isfinite(loss) else 1], device=loss.device, dtype=torch.uint8
        )
        if global_step % 30 == 0:
            flags = [torch.zeros_like(flag) for _ in range(get_world_size())]
            dist.all_gather(flags, flag)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.SUM)

        # If any rank saw NaN/Inf  →  clear grads, skip iteration
        if flag.item():
            logger.info("NaN found in this iteration!")
            optimizer.zero_grad()
            dist.barrier()
            continue                    # go to next batch

            
        # change finishes
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.optimizer.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # logging
        for name in log_names:
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{media_type}-{name}": value})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temperature=model_without_ddp.temp.item())

        if is_main_process() and config.wandb.enable \
                and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix=prefix)

        global_step += 1

        # Step-based checkpoint so a mid-epoch crash doesn't lose all progress.
        # Reads `config.ckpt_save_step_freq` (default 2000); set to 0 to disable.
        ckpt_step_freq = int(getattr(config, "ckpt_save_step_freq", 2000))
        if ckpt_step_freq > 0 and global_step % ckpt_step_freq == 0:
            _save_step_checkpoint(
                model_without_ddp, optimizer, scheduler, scaler,
                config, epoch, global_step,
            )

        if config.debug and global_step % (2 * log_freq + 3) == 0:
            logger.info("debug mode, break training loop")
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
        
    return global_step


@torch.no_grad()
def eval_train(model, eval_loaders, tokenizer, global_step, device, config):
    model.eval()
    avg_eval_accuracy, avg_eval_loss = [], []
    media_types = [loader.dataset.media_type for loader in eval_loaders]
    eval_loader = MetaLoader(name2loader=dict(list(zip(media_types, eval_loaders))))
    for i, batch in enumerate(eval_loader):
        media_type,  (image, text, idx, segment, graph, num_tokens) = batch
        
        image = image.to(device, non_blocking=True)
        if config.vit_type == 'trajvit':
            segment = segment.to(device, non_blocking=True)
            graph = graph.to(device, non_blocking=True)
            num_tokens = num_tokens.to(device, non_blocking=True)
        text_input = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=config.max_txt_l[media_type], return_tensors="pt"
        ).to(device)  # change from "longest" to "max_length"
    
        with torch.no_grad(): loss_dict = model((image, segment, graph, num_tokens), text_input, idx=None)
        avg_eval_accuracy.append(loss_dict["accuracy_ita"])
        avg_eval_loss.append(loss_dict["loss_ita"])
    
    avg_eval_loss = torch.mean(torch.tensor(avg_eval_loss)).item()
    avg_eval_accuracy = torch.mean(torch.tensor(avg_eval_accuracy)).item()


    if is_main_process():
        logger.info(f"Eval Accuracy: {avg_eval_accuracy}, total batch: {i}")
        if config.wandb.enable: 
            log_dict_to_wandb({"video-text matching accuracy": avg_eval_accuracy}, step=global_step, prefix="eval/")
            log_dict_to_wandb({"eval set loss": avg_eval_loss}, step=global_step, prefix="eval/")
    

        


def setup_dataloaders(config, mode="pt", finetune_stage=False):
    
    # train datasets, create a list of data loaders
    logger.info(f"Creating dataset for {mode}")
    train_datasets, collate_fn = create_dataset(f"{mode}_train", config)
    media_types = [d.media_type for d in train_datasets]

    if config.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        samplers = create_sampler(
            train_datasets, [True] * len(media_types), num_tasks, global_rank)
    else:
        samplers = [None] * len(media_types)
    
    collect_collators = []
    for m in media_types:
        collect_collators.append(collate_fn)
    
    
    train_loaders = create_loader(
        train_datasets, samplers,
        batch_size=[config.batch_size[k] for k in media_types],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=collect_collators,
    )  # [0]
    
    if finetune_stage: return train_loaders

    # test datasets, a mapping from dataset name to data loader
    test_datasets, test_collate_fn, test_dataset_names = create_dataset(f"{mode}_eval", config)
    test_loaders = create_loader(
        test_datasets, [None] * len(test_datasets),
        batch_size=[config.batch_size_test[d.media_type] for d in test_datasets],
        num_workers=[config.num_workers_test] * len(test_datasets),
        is_trains=[False] * len(test_datasets),
        collate_fns=[test_collate_fn] * len(test_datasets)
    )
    test_name2loaders = {k: v for k, v in zip(test_dataset_names, test_loaders)}
    
    return train_loaders, test_name2loaders, media_types




def main(config):
    print("is main process?", is_main_process())
    print("global rank", os.environ["RANK"], "local rank", os.environ["LOCAL_RANK"])
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)
        
    logger.info(f"train_file: {config.train_file}")
    logger.info(f"video resolution, {config.image_res}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(config, mode="pt", finetune_stage=False)
    
    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs

    # print("----------------------------")
    # print("step", config.scheduler.num_training_steps, config.scheduler.num_warmup_steps)
    # print("----------------------------")
    # set cudnn.benchmark=True only when input size is fixed
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    cudnn.benchmark = len(train_media_types) == 1
    
    model_cls, train_segmenter_flag, train_vit_flag = identify_model_cls(config)
    
    model, model_without_ddp, optimizer, scheduler, scaler, \
        tokenizer, start_epoch, global_step = setup_model(
            config,
            model_cls=model_cls,
            has_decoder=False,
            pretrain=True,
            find_unused_parameters=True,
        )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    logger.info("Start training")
    start_time = time.time()
    

    for epoch in range(start_epoch, config.scheduler.epochs - config.scheduler.finetune_epochs):

        global_step = train(
            model, train_loaders, optimizer, tokenizer, epoch, global_step,
            device, scheduler, scaler, config, train_segmenter=train_segmenter_flag
        )
        
        dist.barrier()
        
        save_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "epoch": epoch,
                "global_step": global_step,
        }
        torch.save(save_obj, join(config.output_dir, f"latest.pth"))
        
        if is_main_process() and epoch!=0 and (epoch % config.save_freq == 0 or epoch == config.scheduler.epochs-1) :
            torch.save(save_obj, join(config.output_dir, f"ckpt_{epoch:02d}.pth"))
        
        if train_vit_flag:

            for latent_level in range(0, config.traj_model.total_latent_level+1):
                with torch.cuda.amp.autocast(enabled=config.fp16):
                    eval_res = {}
                    for test_name, test_loader in test_name2loaders.items():
                        # Eval JSONs often reference stale data paths (old weka
                        # mount layout). One missing dataset shouldn't kill the
                        # training run — log and skip.
                        try:
                            res = evaluation_wrapper(
                                model_without_ddp, test_loader, tokenizer, device, config, prefix=test_name, latent_level=latent_level)
                            eval_res.update(res)
                        except Exception as e:
                            logger.warning(
                                f"eval failed for test_name={test_name} "
                                f"latent_level={latent_level}: {type(e).__name__}: {e}"
                            )
                            if dist.is_initialized():
                                dist.barrier()  # keep ranks in lockstep after failure
                            continue

                    if is_main_process():
                        if config.wandb.enable:
                            for p, v in eval_res.items():
                                log_dict_to_wandb(v, step=global_step, prefix=p)

                        eval_res = pd.DataFrame(eval_res)
                        logger.info(f"Epoch {epoch}")
                        logger.info(f"\n{eval_res.transpose()}")
                
                    
        dist.barrier()
        
        
        # if epoch % config.eval_freq == 0 or epoch == config.scheduler.epochs-1:   
        #     eval_train(model, eval_loaders, tokenizer, global_step, device, config)
        #     dist.barrier()
                

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()

if __name__ == "__main__":
    cfg = setup_main()
    main(cfg)
    
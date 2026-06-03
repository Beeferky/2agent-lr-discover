import os
import time
import json
import random
import argparse
import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from schedulefree import AdamWScheduleFree
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM
from transformers import default_data_collator
import datasets
import datasets.distributed
import wandb
from tqdm import tqdm
from loguru import logger
from pretraining_utils import training_utils, args_utils
from pretraining_utils.dataloader import PreprocessedIterableDataset
import datetime, pdb, pickle
from torch.profiler import profile, ProfilerActivity

transformers.logging.set_verbosity_error()


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type", type=str, default="llama", choices=["llama"]
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="cola")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--offline_mode", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adamw",
        choices=["adamw", "schedule_free"],
        help="adamw=use training_utils.build_optimizer; schedule_free=use AdamWScheduleFree (no scheduler, no agent LR control)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_restarts"],
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=10_000,
        help="Number of **update steps** to train for. "
        "Notice that gradient accumulation is taken into account.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=training_utils.max_train_tokens_to_number,
        default=None,
        help="Number of tokens to train on. Overwrites num_training_steps. "
        "You can use M and B suffixes, e.g. 100M or 1B.",
    )
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    # baseline mode: scheduler controls full training, no agent LR override
    parser.add_argument("--no_agent", default=False, action="store_true")

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(
    model,
    preprocess_batched,
    pad_idx,
    global_rank,
    world_size,
    device,
    batch_size,
    dataloader=None,
):
    _time = time.time()
    if dataloader is None:
        val_data = datasets.load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        )  # DGX
        val_data = val_data.shuffle(seed=42)
        if is_main_process():
            logger.info(
                f"Loaded validation dataset in {time.time() - _time:.2f} seconds"
            )

        if not args.single_gpu:
            val_data = datasets.distributed.split_dataset_by_node(
                val_data, rank=global_rank, world_size=world_size
            )

        val_data_mapped = val_data.map(
            preprocess_batched,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        )
        val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(
            val_data_mapped, batch_size
        )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    if is_main_process():
        logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in (
        val_data_mapped.batch(batch_size=batch_size)
        if dataloader is None
        else dataloader
    ):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = (
            batch["input_ids"].clone() if "labels" not in batch else batch["labels"]
        )
        batch["labels"][batch["labels"] == pad_idx] = -100

        loss = model(**batch).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens

def save_state_for_agent(step, loss, total_steps):
    """rank 0 only: 原子写训练状态供 agent 读取"""
    state = {
        "step": step,
        "progress_percent": step / total_steps * 100,
        "loss": loss,
    }
    with open("training_state_temp.json", "w") as f:
        json.dump(state, f)
    os.replace("training_state_temp.json", "training_state.json")


def check_for_lr_update(optimizer, last_lr, global_rank, device):
    """rank 0 读 lr_command.txt，broadcast 给所有 rank，返回当前生效 LR"""
    new_lr = last_lr
    if global_rank == 0:
        try:
            with open("lr_command.txt") as f:
                new_lr = float(f.read().strip())
        except (FileNotFoundError, ValueError):
            pass

    lr_tensor = torch.tensor(new_lr, dtype=torch.float32).to(device)
    dist.broadcast(lr_tensor, src=0)
    new_lr = lr_tensor.item()

    if new_lr != last_lr:
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        if global_rank == 0:
            logger.info(f"[Agent] LR 更新: {last_lr:.4e} → {new_lr:.4e}")
    return new_lr


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID")))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(
        f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
    )

    dist.init_process_group(
        backend="nccl",
        rank=global_rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=3600),
    )

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert (
                args.total_batch_size % world_size == 0
            ), "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (
                args.batch_size * world_size
            )
            if is_main_process():
                logger.info(
                    f"{args.gradient_accumulation}-{world_size}-{args.total_batch_size}-{args.batch_size}"
                )
            assert (
                args.gradient_accumulation > 0
            ), "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0:
        logger.remove()

    if global_rank == 0:
        model_name = args.model_config.split("/")[1]
        model_name = model_name.split(".")[0]
        # run_name = (
        #     f"{model_name}-{args.peft_model}"
        #     if args.run_name is None
        #     else args.run_name
        # )
        run_name = args.run_name if args.run_name is not None else model_name

        # Resume the same wandb run across crashes/restarts.
        # agent.py writes {save_dir}/wandb.json containing the run id; reuse it.
        prev_wandb_id = None
        wandb_meta_path = os.path.join(args.save_dir, "wandb.json") if args.save_dir else None
        if wandb_meta_path and os.path.exists(wandb_meta_path):
            try:
                with open(wandb_meta_path) as _f:
                    prev_wandb_id = json.load(_f).get("wandb_id")
            except Exception as _e:
                logger.warning(f"Could not read previous wandb id from {wandb_meta_path}: {_e}")

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            id=prev_wandb_id,
            resume="allow" if prev_wandb_id else None,
        )

        # ★ 自定义 step metric：让所有指标按 update_step 索引，
        # 而不是 wandb 内部的全局递增 step。
        # 这样 rollback 后续跑时，新数据会用相同的 update_step **覆盖** 旧数据，
        # 而不是产生重叠的"双线"。
        wandb.define_metric("update_step")
        wandb.define_metric("train/*",      step_metric="update_step")
        wandb.define_metric("eval/*",       step_metric="update_step")
        wandb.define_metric("lr",           step_metric="update_step")
        wandb.define_metric("tokens_seen",  step_metric="update_step")
        wandb.define_metric("tokens_per_sec", step_metric="update_step")
        wandb.define_metric("throughput_*", step_metric="update_step")
        wandb.define_metric("gradnorm",     step_metric="update_step")
        wandb.define_metric("max_memory",   step_metric="update_step")
        wandb.define_metric("final_eval_*", step_metric="update_step")

        if prev_wandb_id:
            logger.info(f"[wandb] resumed run id={prev_wandb_id} (metrics indexed by update_step → rollback 不会留脏线)")
        else:
            logger.info(f"[wandb] new run id={wandb.run.id}")
            # Persist immediately so the very first crash is still resumable.
            if args.save_dir:
                try:
                    os.makedirs(args.save_dir, exist_ok=True)
                    with open(os.path.join(args.save_dir, "wandb.json"), "w") as _f:
                        json.dump({"wandb_id": wandb.run.id}, _f)
                except OSError as _e:
                    logger.error(
                        f"[wandb] cannot write {args.save_dir}/wandb.json: {_e}. "
                        f"Resume across crashes will NOT work — fix --save_dir to a writable path."
                    )

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    if args.offline_mode:
        logger.info("Loading tokenized data from disk")
        data = datasets.load_from_disk("/datasets/c4/tokenized")
        logger.info("Finished loading from disk")
    else:
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

        seed_for_shuffle = 42
        logger.info(f"Shuffling data with seed {seed_for_shuffle}")
        data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)

    if not args.single_gpu:
        if args.offline_mode:
            train_data: datasets.Dataset = data["train"]
            train_data = datasets.distributed.split_dataset_by_node(
                train_data,
                rank=global_rank,
                world_size=world_size,
            )
            eval_data = data["validation"]
            eval_data = datasets.distributed.split_dataset_by_node(
                eval_data,
                rank=global_rank,
                world_size=world_size,
            )
        else:
            data = datasets.distributed.split_dataset_by_node(
                data,
                rank=global_rank,
                world_size=world_size,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=args.max_length
    )

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    if args.offline_mode:
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=default_data_collator,
            shuffle=True,
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=default_data_collator,
            shuffle=True,
        )
    else:
        # it doesn't matter which tokenizer we use, because we train from scratch
        # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice

        dataset = PreprocessedIterableDataset(
            data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=args.workers
        )
        eval_dataloader = None

    if args.continue_from is not None:

        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")

        model_config = AutoConfig.from_pretrained(args.continue_from)
        model = AutoModelForCausalLM.from_pretrained(
            args.continue_from, torch_dtype=torch.bfloat16
        ).to(device=device)

        logger.info(f"Model successfully loaded")
    else:
        logger.warning(
            f"Did not find training state in {args.continue_from}, global step will start from zero"
        )
        logger.info("*" * 40)

        model_config = AutoConfig.from_pretrained(args.model_config)
        model = AutoModelForCausalLM.from_config(model_config)
        
        if args.dtype in ["bf16", "bfloat16"]:
            model = model.to(device=device, dtype=torch.bfloat16)
        else:
            model = model.to(device=device)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
        

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    # ====== starting config ======= #

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    layer_wise_flag = True if "per_layer" in args.optimizer.lower() else False

    if args.optimizer_type == "schedule_free":
        optimizer = AdamWScheduleFree(
            trainable_params,
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
        )
        agent_status = "active (combined mode)" if not args.no_agent else "disabled"
        logger.info(f"Using AdamWScheduleFree (lr={args.lr}, warmup={args.warmup_steps}). Scheduler disabled; agent LR control: {agent_status}.")
    else:
        optimizer = training_utils.build_optimizer(model, trainable_params, args)
    if layer_wise_flag:
        if not isinstance(optimizer, dict):
            raise ValueError("Layer-wise optimizer is not properly constructed.")

    agent_lr = args.lr  # 追踪 agent 当前生效的 LR

    scheduler = None
    if not layer_wise_flag and args.optimizer_type != "schedule_free":
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if args.continue_from is not None:
        optimizer_checkpoint = torch.load(
            os.path.join(args.continue_from, "optimizer.pt"), map_location="cpu"
        )
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        logger.info(f"Optimizer and scheduler restored from {args.continue_from}")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}"
            )
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(
                f"Will train for {args.num_training_steps - update_step} update steps"
            )

    scheduler_start_step = update_step

    # print params and trainable params
    logger.info(f"Running with {args.model_type}\n")
    logger.info(f"\n{model}\n")
    logger.info(
        f"All params: \n{[n for n,p in model.named_parameters() if p.requires_grad]}\n"
    )
    logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M"
    )
    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    )
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update(
        {
            "max_lr": run_config.pop(
                "lr"
            ),  # rename lr to max_lr to avoid conflicts with scheduler
            "total_params_M": n_total_params / 1_000_000,
            "dataset": "allenai/c4",
            "model": model_config.to_dict(),
            "world_size": world_size,
            "device": str(device),
        }
    )

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")  # save current script
        pbar = tqdm(
            total=args.num_training_steps - update_step, desc="Update steps", ncols=80
        )

    if not args.single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # ##############################

    max_memory = torch.cuda.max_memory_allocated()
    if global_rank == 0:
        logger.info(f"Maximum memory allocated before training: {max_memory} bytes\n")
    torch.cuda.reset_peak_memory_stats()

    if args.optimizer_type == "schedule_free":
        optimizer.train()

    # pdb.set_trace()
    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx // args.gradient_accumulation < update_step:
            # Skipping data that are already seen in previous steps
            continue

        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(
                f"Reached max number of update steps (f{args.num_training_steps}). Stopping training."
            )
            print(f"Rank {global_rank} stopping training.")
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = (
            batch["input_ids"].clone() if "labels" not in batch else batch["labels"]
        )
        batch["labels"][batch["labels"] == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch).loss
        scaled_loss = loss / args.gradient_accumulation

        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        #######
        if args.grad_clipping != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        grad_norm = sum(
            [
                torch.norm(p.grad.clone().detach().cpu())
                for p in model.parameters()
                if p.grad is not None
            ]
        )

        if global_rank == 0:
            pbar.update(1)

        if not layer_wise_flag:
            optimizer.step()
            # Agent 模式下 scheduler 只管 warmup，warmup 后由 agent 接管；
            # --no_agent 基线模式下 scheduler 全程接管。
            # Schedule-Free: optimizer 内部自管 LR/warmup，跳过 scheduler。
            if scheduler is not None and (args.no_agent or update_step < args.warmup_steps):
                scheduler.step()
            optimizer.zero_grad()

        # Agent LR 指令检查（所有 rank 同步）——基线模式下跳过
        # Combined mode (agent + SF): 允许 agent 改 SF 的 base lr; SF 内部仍做 averaging
        # 注意 lr_max 在 SF 内只升不降，agent 提高 lr 会拉高 lr_max 并改变 averaging 权重
        if not args.no_agent:
            agent_lr = check_for_lr_update(optimizer, agent_lr, global_rank, device)

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if (
            local_step > args.gradient_accumulation
            and update_step % args.save_every == 0
            and global_rank == 0
        ):
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(
                f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
            )
            os.makedirs(args.save_dir, exist_ok=True)
            # Schedule-Free: ckpt 必须用 averaged weights（deployable），
            # 否则保存的是 training weights，部署 ppl 会差。
            if args.optimizer_type == "schedule_free":
                optimizer.eval()
            model.module.save_pretrained(
                current_model_directory, max_shard_size="100GB"
            )
            if args.optimizer_type == "schedule_free":
                optimizer.train()

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # evaluation (also run an early eval at step 200 for quick feedback)
        do_eval = (update_step % args.eval_every == 0) or (update_step == 200)
        if do_eval:
            logger.info(f"Performing evaluation at step {update_step}")
            # Schedule-Free: eval 必须用 averaged weights，否则 ppl 会差很多
            if args.optimizer_type == "schedule_free":
                optimizer.eval()
            model.eval()
            eval_t0 = time.time()
            try:
                total_loss, evaluated_on_tokens = evaluate_model(
                    model,
                    preprocess_batched,
                    pad_idx,
                    global_rank,
                    world_size,
                    device,
                    args.batch_size,
                    eval_dataloader,
                )
                eval_dur = time.time() - eval_t0
                eval_ppl = float(np.exp(min(float(total_loss), 30.0)))
                if global_rank == 0:
                    wandb.log(
                        {
                            "update_step":   update_step,           # ← x 轴字段
                            "eval/loss":     float(total_loss),
                            "eval/ppl":      eval_ppl,
                            "eval/bpb":      float(total_loss) / math.log(2),
                            "eval/tokens":   evaluated_on_tokens,
                            "eval/eval_time_sec": eval_dur,
                        },
                    )
                logger.info(
                    f"Eval @ step {update_step}: loss={float(total_loss):.4f} ppl={eval_ppl:.2f} ({eval_dur:.1f}s)"
                )
            except Exception as e:
                logger.exception(f"[Eval] failed at step {update_step}: {e}")
            model.train()
            if args.optimizer_type == "schedule_free":
                optimizer.train()

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        max_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        if global_rank == 0:
            wandb_payload = {
                "update_step":         update_step,             # ← x 轴字段
                "train/loss":          loss.item(),
                "lr":                  lr,
                "agent/qwen_target_lr": agent_lr,                # Qwen 通过 lr_command.txt 想要的 LR
                "tokens_seen":         tokens_seen,
                "tokens_per_sec":      tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches":  batches_in_update / update_time,
                "gradnorm":            grad_norm,
                "max_memory":          max_memory,
            }
            # Schedule-Free 内部状态：lr_max (ratchet) 与 scheduled_lr (实际生效 LR)
            if args.optimizer_type == "schedule_free" and not layer_wise_flag:
                pg0 = optimizer.param_groups[0]
                wandb_payload["sf/lr_max"]       = pg0.get("lr_max", 0.0)
                wandb_payload["sf/scheduled_lr"] = pg0.get("scheduled_lr", 0.0)
                wandb_payload["sf/weight_sum"]   = pg0.get("weight_sum", 0.0)
            wandb.log(wandb_payload)

        # 每 100 steps 写训练状态供 agent 读取
        if global_rank == 0 and update_step % 100 == 0:
            save_state_for_agent(update_step, loss.item(), args.num_training_steps)

        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0:
        pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(
            f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
        )
        os.makedirs(args.save_dir, exist_ok=True)

        if args.optimizer_type == "schedule_free":
            optimizer.eval()
        model.module.save_pretrained(current_model_directory)
        if args.optimizer_type == "schedule_free":
            optimizer.train()

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    # Schedule-Free: 用 averaged weights 跑 final eval（deployable ppl）
    if args.optimizer_type == "schedule_free":
        optimizer.eval()
    model.eval()
    del loss, optimizer, scheduler
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model,
        preprocess_batched,
        pad_idx,
        global_rank,
        world_size,
        device,
        args.batch_size,
        eval_dataloader,
    )

    if global_rank == 0:
        final_ppl = float(np.exp(total_loss))
        wandb.log(
            {
                "update_step":     update_step,                     # ← x 轴字段
                "final_eval_loss": total_loss,
                "final_eval_ppl":  final_ppl,                       # 与 eval/ppl 命名一致
                "final_eval_tokens": evaluated_on_tokens,
            },
        )
        logger.info(
            f"Eval loss and perplexity at step {update_step}: {total_loss}, {final_ppl}"
        )
        # 写本地 final_result.json 供 agent.py 读取（跨 run 经验累积）
        if args.save_dir:
            try:
                os.makedirs(args.save_dir, exist_ok=True)
                with open(os.path.join(args.save_dir, "final_result.json"), "w") as _f:
                    json.dump({
                        "final_eval_loss": float(total_loss),
                        "final_eval_ppl": final_ppl,
                        "final_eval_tokens": int(evaluated_on_tokens),
                        "update_step": int(update_step),
                        "seed": int(args.seed),
                    }, _f, indent=2)
            except OSError as _e:
                logger.warning(f"Could not write final_result.json: {_e}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)

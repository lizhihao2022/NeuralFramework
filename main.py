# main.py
import torch
import torch.distributed as dist

from config import parser
from utils import set_up_logger, set_seed, load_config, save_config, init_distributed
from trainers import TRAINER_REGISTRY


def main():
    # ========= Step 0. Optional: initialize distributed =========
    distributed, local_rank = init_distributed()

    # ========= Step 1. Parse CLI args and load config =========
    args = parser.parse_args()
    args = vars(args)
    args = load_config(args)  # merge CLI args with YAML config

    # Ensure "train" sub-dict exists
    train_cfg = args.setdefault("train", {})

    if distributed:
        train_cfg["local_rank"] = local_rank
        train_cfg["world_size"] = dist.get_world_size()
        train_cfg["rank"] = dist.get_rank()
    else:
        train_cfg["local_rank"] = 0
        train_cfg["world_size"] = 1
        train_cfg["rank"] = 0

    # ========= Step 2. Setup logger and save config =========
    if distributed:
        # Only rank 0 creates logger and writes config to disk
        if train_cfg["rank"] == 0:
            saving_path, saving_name = set_up_logger(args)
            save_config(args, saving_path)
        else:
            saving_path, saving_name = None, None

        # Broadcast saving_path and saving_name to all ranks
        payload = [saving_path, saving_name]
        dist.broadcast_object_list(payload, src=0)
        saving_path, saving_name = payload
    else:
        # Single-process: just create logger and save config
        saving_path, saving_name = set_up_logger(args)
        save_config(args, saving_path)

    train_cfg["saving_path"] = saving_path
    train_cfg["saving_name"] = saving_name

    # ========= Step 3. Set random seed and (optionally) device =========
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    if distributed:
        # local_rank has already been set in init_distributed()
        torch.cuda.set_device(local_rank)
    # For single-GPU or CPU setups, you can handle device logic inside trainer

    # ========= Step 4. Build trainer (creates model, dataloader, etc.) =========
    trainer_name = args["model"]["name"]
    trainer_cls = TRAINER_REGISTRY[trainer_name]
    trainer = trainer_cls(args)

    # ========= Step 5. Run training / evaluation loop =========
    trainer.process()

    # ========= Step 6. Clean up distributed environment =========
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

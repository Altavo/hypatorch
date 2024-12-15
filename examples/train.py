import hydra
from hydra.utils import instantiate
import hypatorch
import os
import sys
import logging

def get_rank():
    return int(os.environ.get("RANK", "-1"))

def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def hypatorch_train(cfg):
    rank = get_rank()
    world_size = get_world_size()

    logger = None
    if rank <= 0:
        logger = instantiate(cfg.logger)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    model = instantiate(cfg.model)
    dataset = instantiate(cfg.dataset)
    trainer = hypatorch.Trainer(**cfg.trainer, rank=rank, world_size=world_size)

    trainer.train(model=model, train_dataset=dataset.train, val_dataset=dataset.val, loader_args=cfg.dataloader, logger=logger)

if __name__ == "__main__":

    # Add +data_root to the command line arguments
    sys.argv.append(f"+data_root={os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_root')}")

    # Disable hydra output dir creation on non rank 0
    if get_rank() > 0:
        sys.argv.append("hydra.output_subdir=null")

    hypatorch_train()



import hydra
from hydra.utils import instantiate
import hypatorch
import os
import sys
import torch.multiprocessing as mp

def train(rank, cfg):
    logger = None
    if rank == 0:
        logger = instantiate(cfg.logger)

    model = instantiate(cfg.model)
    dataset = instantiate(cfg.dataset)
    trainer = hypatorch.Trainer(**cfg.trainer, rank=rank)

    trainer.train(model=model, train_dataset=dataset.train, val_dataset=dataset.val, loader_args=cfg.dataloader, logger=logger)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def hypatorch_train(cfg):

    mp.spawn(train,
             args=(cfg,),
             nprocs=cfg.trainer.world_size,
             join=True)

if __name__ == "__main__":

    # Add +data_root to the command line arguments
    sys.argv.append(f"+data_root={os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_root')}")
    hypatorch_train()



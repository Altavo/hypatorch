import hydra
from hydra.utils import instantiate
import hypatorch
import os
import sys

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def hypatorch_train(cfg):
    logger = instantiate(cfg.logger)
    model = instantiate(cfg.model)

    dataset = instantiate(cfg.dataset)

    trainer = hypatorch.Trainer(**cfg.trainer)

    trainer.train(model=model, train_dataset=dataset.train, val_dataset=dataset.val, loader_args=cfg.dataloader, logger=logger)

if __name__ == "__main__":

    # Add +data_root to the command line arguments
    sys.argv.append(f"+data_root={os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data_root')}")
    hypatorch_train()



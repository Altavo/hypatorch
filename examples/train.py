from hydra import initialize, compose
from hydra.utils import instantiate
import os
import hypatorch


if __name__ == "__main__":
    rel_config_dir = os.path.relpath(os.path.dirname(__file__), os.getcwd())
    with initialize(
        config_path=os.path.join(rel_config_dir, "conf"), version_base="1.1"
    ):
        cfg = compose(config_name="config.yaml", overrides=['+experiment=mnist_lenet'])


    logger = instantiate(cfg.logger)
    model = instantiate(cfg.model)
    dataset = instantiate(cfg.dataset)

    trainer = hypatorch.Trainer(**cfg.trainer)

    trainer.train(model=model, train_dataset=dataset.train, val_dataset=dataset.val, loader_args=cfg.dataloader, max_epochs=2, logger=logger)


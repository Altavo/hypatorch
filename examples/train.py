from hydra import initialize, compose
from hydra.utils import instantiate
import os
import hypatorch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MNIST Training")
    parser.add_argument("--model", type=str, default="lenet")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--autocast_dtype", type=str, default=None)
    args = parser.parse_args()

    rel_config_dir = os.path.relpath(os.path.dirname(__file__) or os.getcwd(), os.getcwd())
    with initialize(
        config_path=os.path.join(rel_config_dir, "conf"), version_base="1.1"
    ):
        overrides = ["+experiment=mnist", f"+model={args.model}", f"device={args.device}"]
        if args.compile_model:
            overrides.append("+trainer.compile_model=True")
        if args.autocast_dtype:
            overrides.append(f"+trainer.autocast_dtype={args.autocast_dtype}")

        cfg = compose(config_name="config.yaml", overrides=overrides)


    logger = instantiate(cfg.logger)
    model = instantiate(cfg.model)
    dataset = instantiate(cfg.dataset)

    trainer = hypatorch.Trainer(**cfg.trainer)

    trainer.train(model=model, train_dataset=dataset.train, val_dataset=dataset.val, loader_args=cfg.dataloader, logger=logger)


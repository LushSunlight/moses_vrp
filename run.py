import os, sys
import numpy as np
os.environ["MY_RANDOM_NUM"] = str(int(round(np.random.rand(), 5)*1e5))

from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
import torch.serialization
import wandb
import math

from lightning import Callback, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
import omegaconf
from rl4co import utils
from rl4co.utils import RL4COTrainer
from envs.mtvrp.env import MTVRPEnv

#OmegaConf.register_new_resolver('random', lambda x: str(int(round(np.random.rand(), x)*1e5)))
pyrootutils.setup_root(__file__, indicator="run.py", pythonpath=True)
log = utils.get_pylogger(__name__)

# Required for loading old checkpoints under PyTorch 2.6+ weights-only behavior.
torch.serialization.add_safe_globals([MTVRPEnv])

# Prefer WANDB_API_KEY from environment; fallback to normal wandb login behavior.
wandb_api_key = os.getenv("WANDB_API_KEY", "").strip()
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    wandb.login()


@utils.task_wrapper
def run(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.
    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # We instantiate the environment separately and then pass it to the model
    log.info(f"Instantiating environment <{cfg.env._target_}>")
    env = hydra.utils.instantiate(cfg.env)

    # Note that the RL environment is instantiated inside the model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, env)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"), model=model)

    log.info("Instantiating trainer...")
    trainer: RL4COTrainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile", False):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        fit_kwargs = {"model": model, "ckpt_path": cfg.get("ckpt_path")}
        if cfg.get("ckpt_path"):
            # PyTorch 2.6 defaults to weights_only=True, which can block full checkpoint restore.
            fit_kwargs["weights_only"] = False
        try:
            trainer.fit(**fit_kwargs)
        except TypeError:
            # Backward compatibility for Lightning versions without weights_only in fit().
            fit_kwargs.pop("weights_only", None)
            trainer.fit(**fit_kwargs)

        train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict



@hydra.main(version_base="1.3", config_path="configs", config_name="main.yaml")
def train(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = run(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    train()
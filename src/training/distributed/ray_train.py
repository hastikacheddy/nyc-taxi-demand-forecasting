"""
Production distributed training — Ray Train (TorchTrainer).

The same DDP training, but orchestrated by Ray, which is the "Controller → Ray
Cluster → Worker 1/2/3 GPU" topology from the design sketch. Ray handles cluster
scheduling, worker placement, checkpoint management, and fault tolerance
declaratively — you describe the scale and the failure budget, not the plumbing.

    python -m src.training.distributed.ray_train --workers 3 --use-gpu

Why Ray in addition to DDP:
  * `ScalingConfig(num_workers, use_gpu)` — scale is a config value, and the same
    script runs on 1 laptop GPU or a 3-node A100 cluster.
  * `FailureConfig(max_failures)` — declarative failure recovery: a dead worker
    triggers a restart from the latest Ray checkpoint, no bespoke retry code.
  * Ray Train composes with Ray Serve (ADR-003) — one substrate for distributed
    training *and* serving, which is why it's a first-class option here.

Optional dependency: ray[train] + torch. `pip install "ray[train]" torch`.
"""
from __future__ import annotations

import argparse
import os
import tempfile

try:
    import torch
    import torch.nn as nn
    import ray.train
    from ray.train import ScalingConfig, RunConfig, FailureConfig, Checkpoint
    from ray.train.torch import TorchTrainer, prepare_model, prepare_data_loader
    from torch.utils.data import DataLoader, TensorDataset
    _RAY = True
except ImportError:
    _RAY = False


def _forecaster(n_features: int):
    return nn.Sequential(
        nn.Linear(n_features, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )


def _synthetic(n=20000, d=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g)
    y = X @ w + 0.1 * torch.randn(n, 1, generator=g) + 2.0
    return TensorDataset(X, y)


def train_loop_per_worker(cfg: dict) -> None:
    """Runs on EACH Ray worker. Ray's prepare_* helpers inject DDP wrapping and
    per-worker data sharding, so this reads like single-process code."""
    epochs = cfg["epochs"]
    dataset = _synthetic()
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)
    loader = prepare_data_loader(loader)                 # shards across workers

    model = prepare_model(_forecaster(dataset.tensors[0].shape[1]))  # DDP wrap
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    # Failure recovery: resume from the checkpoint Ray hands back after a restart.
    start_epoch = 0
    ckpt = ray.train.get_checkpoint()
    if ckpt:
        with ckpt.as_directory() as d:
            # weights_only=True → safe checkpoint load (no code execution on load)
            state = torch.load(os.path.join(d, "state.pt"), weights_only=True)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optim"])
            start_epoch = state["epoch"] + 1

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        for X, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()          # Ray/DDP all-reduce
            optimizer.step()
            running += loss.item()

        # Report metric + checkpoint to the controller (rank 0 writes the file).
        with tempfile.TemporaryDirectory() as d:
            if ray.train.get_context().get_world_rank() == 0:
                torch.save({"epoch": epoch,
                            "model": model.state_dict(),
                            "optim": optimizer.state_dict()},
                           os.path.join(d, "state.pt"))
            ray.train.report(
                {"loss": running / len(loader), "epoch": epoch},
                checkpoint=Checkpoint.from_directory(d),
            )


def run(workers: int = 3, use_gpu: bool = False, epochs: int = 20) -> None:
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"epochs": epochs, "batch_size": 256, "lr": 1e-3},
        # The topology: N workers, each with (optionally) a GPU.
        scaling_config=ScalingConfig(num_workers=workers, use_gpu=use_gpu),
        # Declarative fault tolerance: tolerate up to 3 worker failures, each
        # recovered from the latest checkpoint.
        run_config=RunConfig(failure_config=FailureConfig(max_failures=3)),
    )
    result = trainer.fit()
    print(f"final loss={result.metrics.get('loss'):.4f}  checkpoint={result.checkpoint}")


def main() -> None:
    if not _RAY:
        raise SystemExit('Ray/torch not installed. `pip install "ray[train]" torch` '
                         "to run, or use data_parallel_demo.py for a dep-free run.")
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    run(args.workers, args.use_gpu, args.epochs)


if __name__ == "__main__":
    main()

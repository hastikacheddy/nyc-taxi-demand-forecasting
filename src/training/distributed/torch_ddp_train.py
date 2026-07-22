"""
Production distributed training — PyTorch DistributedDataParallel (DDP).

The real-GPU equivalent of data_parallel_demo.py: same four ideas (shard,
all-reduce, checkpoint, recover) implemented with the industry-standard stack.
Launch with torchrun, which handles rank assignment and *elastic* restarts:

    torchrun --nproc_per_node=3 --max-restarts=3 \
        -m src.training.distributed.torch_ddp_train --epochs 20

  * `--nproc_per_node=3`  → the "Worker 1/2/3 GPU" fan-out.
  * `--max-restarts=3`    → failure recovery: if a worker dies, torchrun restarts
                            the group and training resumes from the last checkpoint.

Optional dependency: torch. Kept out of requirements.txt so the repo stays light;
`pip install torch` to run. On CPU it uses the gloo backend, on GPU nccl — the
code is identical, which is the point.

Maps to the platform: this job runs on the GPU pool under the
`platform-training` PriorityClass (kubernetes/serving/gpu-priorityclasses.yaml),
so it soaks up idle A100s and is pre-empted the instant online serving needs them
— hence checkpoint frequently.
"""
from __future__ import annotations

import argparse
import os

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    _TORCH = True
except ImportError:  # keeps import-time safe without torch installed
    _TORCH = False


CKPT_PATH = os.environ.get("DDP_CKPT", os.path.join("data", "checkpoints", "ddp.pt"))


def _forecaster(n_features: int):
    """A small MLP forecaster over the taxi features (Volume lags + calendar)."""
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


def _setup():
    """Initialise the process group from torchrun's env vars. nccl on GPU,
    gloo on CPU — chosen automatically."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank


def _save_checkpoint(model, optimizer, epoch, rank):
    """Only rank 0 writes, and it writes atomically — a pre-emption mid-write must
    never corrupt the checkpoint the restart will read."""
    if rank != 0:
        return
    os.makedirs(os.path.dirname(CKPT_PATH) or ".", exist_ok=True)
    tmp = CKPT_PATH + ".tmp"
    torch.save({"epoch": epoch,
                "model": model.module.state_dict(),   # unwrap DDP
                "optim": optimizer.state_dict()}, tmp)
    os.replace(tmp, CKPT_PATH)


def _load_checkpoint(model, optimizer, device):
    """Every rank loads the same checkpoint (map_location keeps it device-correct)
    so all workers resume from an identical state — the recovery guarantee."""
    if not os.path.exists(CKPT_PATH):
        return 0
    # weights_only=True restricts unpickling to tensors/primitives — the secure
    # way to load a checkpoint (no arbitrary code execution on load).
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.module.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    return ckpt["epoch"] + 1


def train(epochs: int = 20, batch_size: int = 256, lr: float = 1e-3) -> None:
    rank, local_rank = _setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dataset = _synthetic()
    # DistributedSampler shards the data so each rank sees a disjoint slice.
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = _forecaster(dataset.tensors[0].shape[1]).to(device)
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    start_epoch = _load_checkpoint(model, optimizer, device)  # resume if a checkpoint exists
    if rank == 0 and start_epoch:
        print(f"[rank0] resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)  # reshuffle deterministically across ranks
        model.train()
        running = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()          # DDP all-reduces gradients across ranks here
            optimizer.step()
            running += loss.item()
        _save_checkpoint(model, optimizer, epoch, rank)
        if rank == 0:
            print(f"[rank0] epoch {epoch}  loss={running/len(loader):.4f}")

    dist.destroy_process_group()


def main() -> None:
    if not _TORCH:
        raise SystemExit("PyTorch not installed. `pip install torch` to run this "
                         "example, or use data_parallel_demo.py for a dep-free run.")
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    train(args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()

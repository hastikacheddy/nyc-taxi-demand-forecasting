"""
Dependency-free data-parallel training — the engineering, without the GPUs.

The point of a distributed-training example is not "I called `torchrun`". It is
demonstrating the four hard parts that actually matter:

  1. **Sharding** the dataset across workers,
  2. **All-reduce**: each worker computes gradients on its shard, gradients are
     averaged, every worker steps with the same update (synchronous SGD),
  3. **Checkpointing** so a job can resume mid-training,
  4. **Failure recovery**: a worker dying does not corrupt or lose the run.

This module implements all four with numpy + a worker pool, so it *runs and is
tested* on a laptop. The production equivalents — `torch_ddp_train.py` (PyTorch
DDP) and `ray_train.py` (Ray Train) — implement the same four ideas on real GPUs;
this file is the executable explanation of what they do under the hood.

Local workers default to threads (numpy releases the GIL on the vectorized math,
and thread vs process vs node is an execution-substrate detail — the *algorithm*
is identical). Set executor="process" to fan out across cores.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ── Model: linear regression, MSE loss (stand-in for the forecaster) ──
@dataclass
class Params:
    w: np.ndarray
    b: float

    def copy(self) -> "Params":
        return Params(self.w.copy(), float(self.b))


def _grad_on_shard(params: Params, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
    """One worker's local gradient + loss on its data shard.

    Returns (grad_w, grad_b, sum_sq_error, n) so the coordinator can average
    correctly by sample count — this is the all-reduce payload."""
    n = len(y)
    if n == 0:
        return np.zeros_like(params.w), 0.0, 0.0, 0
    pred = X @ params.w + params.b
    err = pred - y
    grad_w = (2.0 / n) * (X.T @ err)
    grad_b = (2.0 / n) * float(err.sum())
    return grad_w, grad_b, float((err ** 2).sum()), n


class _FlakyWorker:
    """Wraps the gradient computation to inject deterministic worker failures,
    so failure-recovery is a *tested* behaviour, not a story. `fail_on` is a set
    of (step, worker_id) pairs that should raise once."""

    def __init__(self, fail_on: Optional[set] = None) -> None:
        self.fail_on = fail_on or set()
        self._already_failed = set()

    def __call__(self, args) -> Tuple[np.ndarray, float, float, int]:
        step, wid, params, X, y = args
        key = (step, wid)
        if key in self.fail_on and key not in self._already_failed:
            self._already_failed.add(key)
            raise RuntimeError(f"simulated worker {wid} crash at step {step}")
        return _grad_on_shard(params, X, y)


# ── Checkpointing ─────────────────────────────────────────────────
def save_checkpoint(path: str, params: Params, step: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # np.savez appends .npz unless the name already ends in it — so give the temp
    # file that extension, write, then atomically rename onto the final path. The
    # atomic replace means a crash mid-write never leaves a corrupt checkpoint.
    tmp = path + ".tmp.npz"
    np.savez(tmp, w=params.w, b=params.b, step=step)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> Optional[Tuple[Params, int]]:
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    return Params(d["w"], float(d["b"])), int(d["step"])


# ── The distributed trainer ───────────────────────────────────────
@dataclass
class TrainResult:
    params: Params
    steps_run: int
    final_loss: float
    loss_history: List[float]
    recovered_failures: int
    resumed_from_step: int


class DataParallelTrainer:
    def __init__(
        self,
        n_workers: int = 4,
        lr: float = 0.05,
        executor: str = "thread",
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 25,
        max_worker_retries: int = 2,
    ) -> None:
        self.n_workers = n_workers
        self.lr = lr
        self.executor = executor
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.max_worker_retries = max_worker_retries

    def _shard(self, X: np.ndarray, y: np.ndarray):
        """Contiguous shards, one per worker (a DistributedSampler analog)."""
        idx = np.array_split(np.arange(len(y)), self.n_workers)
        return [(X[i], y[i]) for i in idx]

    def _pool(self):
        cls = ProcessPoolExecutor if self.executor == "process" else ThreadPoolExecutor
        return cls(max_workers=self.n_workers)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        steps: int = 300,
        resume: bool = True,
        _fail_on: Optional[set] = None,
    ) -> TrainResult:
        n_features = X.shape[1]
        params = Params(np.zeros(n_features), 0.0)
        start_step = 0
        resumed_from = 0

        # Failure recovery, part 1: resume the whole job from the last checkpoint.
        if resume and self.checkpoint_path:
            ck = load_checkpoint(self.checkpoint_path)
            if ck is not None:
                params, start_step = ck
                resumed_from = start_step

        shards = self._shard(X, y)
        worker = _FlakyWorker(_fail_on)
        loss_history: List[float] = []
        recovered = 0

        with self._pool() as pool:
            for step in range(start_step, steps):
                # scatter: each worker gets the CURRENT params + its shard
                tasks = [(step, wid, params.copy(), sx, sy)
                         for wid, (sx, sy) in enumerate(shards)]
                grads: List[Tuple[np.ndarray, float, float, int]] = []

                # gather with per-worker retry — a crashed worker's shard is
                # recomputed rather than lost (failure recovery, part 2).
                pending = list(tasks)
                attempts = 0
                while pending and attempts <= self.max_worker_retries:
                    results = []
                    futs = {pool.submit(worker, t): t for t in pending}
                    failed = []
                    for fut, t in futs.items():
                        try:
                            results.append(fut.result())
                        except Exception:
                            failed.append(t)
                    grads.extend(results)
                    if failed:
                        recovered += len(failed)
                    pending = failed
                    attempts += 1
                if pending:
                    raise RuntimeError(f"workers failed past retry budget at step {step}")

                # all-reduce: sample-count-weighted average of gradients + loss
                total_n = sum(g[3] for g in grads) or 1
                gw = sum(g[0] * g[3] for g in grads) / total_n
                gb = sum(g[1] * g[3] for g in grads) / total_n
                mse = sum(g[2] for g in grads) / total_n
                loss_history.append(mse)

                # synchronous SGD step (identical update on the shared params)
                params.w -= self.lr * gw
                params.b -= self.lr * gb

                if self.checkpoint_path and (step + 1) % self.checkpoint_every == 0:
                    save_checkpoint(self.checkpoint_path, params, step + 1)

        final_loss = loss_history[-1] if loss_history else float("nan")
        if self.checkpoint_path:
            save_checkpoint(self.checkpoint_path, params, steps)
        return TrainResult(params, steps - start_step, final_loss, loss_history,
                           recovered, resumed_from)


def _make_data(n=2000, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    true_w = rng.normal(size=d)
    y = X @ true_w + 0.1 * rng.normal(size=n) + 2.0
    return X, y, true_w


if __name__ == "__main__":
    X, y, true_w = _make_data()
    ckpt = os.path.join("data", "checkpoints", "dp_demo.npz")
    trainer = DataParallelTrainer(n_workers=4, checkpoint_path=ckpt, checkpoint_every=25)
    t0 = time.time()
    # inject a worker crash at step 30 to exercise recovery
    res = trainer.train(X, y, steps=200, resume=False, _fail_on={(30, 2)})
    print(f"steps={res.steps_run} final_loss={res.final_loss:.4f} "
          f"recovered_failures={res.recovered_failures} in {time.time()-t0:.2f}s")
    print(f"resume test: re-run picks up from checkpoint step={load_checkpoint(ckpt)[1]}")

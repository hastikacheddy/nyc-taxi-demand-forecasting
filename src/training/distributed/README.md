# Distributed training

Three implementations of the *same* data-parallel training, at three fidelities —
because the engineering (shard → all-reduce → checkpoint → recover) is identical
whether it runs on a laptop or a GPU cluster.

```
                 Controller / launcher
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
    Worker 1          Worker 2          Worker 3
   (data shard)      (data shard)      (data shard)
        └──────── all-reduce gradients ───────┘
                          │
                 synchronous SGD step
                          │
                   checkpoint (atomic)
```

| File | Stack | Deps | Runs here? | Demonstrates |
|---|---|---|---|---|
| [`data_parallel_demo.py`](data_parallel_demo.py) | numpy + worker pool | **none** | ✅ **yes, tested** | shard, all-reduce, checkpoint, failure recovery |
| [`torch_ddp_train.py`](torch_ddp_train.py) | PyTorch DDP + `torchrun` | `torch` | on a GPU box | DDP, DistributedSampler, elastic restart |
| [`ray_train.py`](ray_train.py) | Ray Train `TorchTrainer` | `ray[train]`, `torch` | on a Ray cluster | declarative scale + fault tolerance |

## Run the dependency-free demo (works anywhere)

```bash
python -m src.training.distributed.data_parallel_demo
# steps=200 final_loss=0.0100 recovered_failures=1 ...   (converges, recovers, resumes)

pytest tests/test_distributed_training.py     # 5 tests incl. all-reduce correctness
```

The demo's headline test asserts that **4 workers produce byte-identical
parameters to 1 worker** (to 1e-10) — synchronous sample-weighted gradient
averaging *is* full-batch gradient descent. That equivalence is the correctness
core of every data-parallel trainer; proving it on numpy is proving it for DDP.

## Run the production versions

```bash
# PyTorch DDP — 3 workers, tolerate 3 restarts (failure recovery)
pip install torch
torchrun --nproc_per_node=3 --max-restarts=3 -m src.training.distributed.torch_ddp_train

# Ray Train — same job, declarative scale + FailureConfig
pip install "ray[train]" torch
python -m src.training.distributed.ray_train --workers 3 --use-gpu
```

## The four properties, and where each is proven

| Property | Dep-free demo | DDP | Ray |
|---|---|---|---|
| **Sharding** | `np.array_split` per worker | `DistributedSampler` | `prepare_data_loader` |
| **All-reduce** | sample-weighted grad average (tested ≡ 1 worker) | `loss.backward()` under DDP | Ray/DDP under the hood |
| **Checkpointing** | atomic `.npz` every K steps | atomic `torch.save` on rank 0 | `ray.train.report(checkpoint=…)` |
| **Failure recovery** | per-worker retry + resume-from-checkpoint | `torchrun --max-restarts` | `FailureConfig(max_failures=3)` |

## Platform integration

These jobs run on the **GPU pool** under the `platform-training` PriorityClass
([`kubernetes/serving/gpu-priorityclasses.yaml`](../../../kubernetes/serving/gpu-priorityclasses.yaml)),
which is globally lowest and `preemptionPolicy: Never` — so training soaks up idle
A100s for free and is evicted the instant online serving needs a GPU
([GPU_DESIGN §2](../../../docs/platform/GPU_DESIGN.md), [COST_MODEL §4](../../../docs/platform/COST_MODEL.md)).
That pre-emptibility is exactly *why* checkpoint frequency is a first-class knob
here. On Kubernetes, the Ray path is launched as a KubeRay `RayJob`.

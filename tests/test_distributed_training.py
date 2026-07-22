"""
Tests for the dependency-free data-parallel trainer
(src/training/distributed/data_parallel_demo.py).

These prove the four properties a distributed-training example must actually
demonstrate: correct all-reduce (N workers == 1 worker), convergence,
checkpoint/resume, and worker-failure recovery.
"""
import os

import numpy as np
import pytest

from src.training.distributed.data_parallel_demo import (
    DataParallelTrainer, _make_data, load_checkpoint,
)


def test_converges():
    X, y, _ = _make_data(seed=1)
    res = DataParallelTrainer(n_workers=4, lr=0.05).train(X, y, steps=300, resume=False)
    # data noise floor is 0.1^2 = 0.01; a converged MSE should be near it
    assert res.final_loss < 0.02
    assert res.loss_history[0] > res.loss_history[-1] * 5   # substantial drop


def test_all_reduce_equals_single_worker():
    """Synchronous, sample-weighted data-parallel SGD is mathematically identical
    to single-worker full-batch GD. 4 workers must match 1 worker to numerical
    precision — the core correctness guarantee of the all-reduce."""
    X, y, _ = _make_data(seed=2)
    one = DataParallelTrainer(n_workers=1, lr=0.05).train(X, y, steps=150, resume=False)
    four = DataParallelTrainer(n_workers=4, lr=0.05).train(X, y, steps=150, resume=False)
    assert np.allclose(one.params.w, four.params.w, atol=1e-10)
    assert abs(one.params.b - four.params.b) < 1e-10


def test_checkpoint_and_resume(tmp_path):
    X, y, _ = _make_data(seed=3)
    ckpt = os.path.join(tmp_path, "dp.npz")

    # phase 1: train 100 steps, checkpoint written
    t1 = DataParallelTrainer(n_workers=4, lr=0.05, checkpoint_path=ckpt, checkpoint_every=25)
    t1.train(X, y, steps=100, resume=False)
    saved = load_checkpoint(ckpt)
    assert saved is not None and saved[1] == 100

    # phase 2: a fresh trainer resumes and finishes to 200
    t2 = DataParallelTrainer(n_workers=4, lr=0.05, checkpoint_path=ckpt, checkpoint_every=25)
    res = t2.train(X, y, steps=200, resume=True)
    assert res.resumed_from_step == 100      # picked up where phase 1 left off
    assert res.steps_run == 100              # only ran the remaining 100
    assert res.final_loss < 0.02             # and still converged


def test_worker_failure_is_recovered():
    X, y, _ = _make_data(seed=4)
    trainer = DataParallelTrainer(n_workers=4, lr=0.05, max_worker_retries=2)
    # crash worker 2 at step 10 and worker 0 at step 20 — both should be retried
    res = trainer.train(X, y, steps=120, resume=False, _fail_on={(10, 2), (20, 0)})
    assert res.recovered_failures == 2
    assert res.final_loss < 0.02             # recovery didn't corrupt the run


def test_failure_past_retry_budget_raises():
    X, y, _ = _make_data(seed=5)
    # worker keeps failing but budget is 0 retries → the job must surface the error
    trainer = DataParallelTrainer(n_workers=2, lr=0.05, max_worker_retries=0)
    with pytest.raises(RuntimeError):
        trainer.train(X, y, steps=50, resume=False, _fail_on={(5, 1)})

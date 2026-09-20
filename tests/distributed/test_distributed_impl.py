"""Distributed communicator plugin lifecycle + PyNCCL wrapper guards.

Runs on CPU: TorchDistributedImpl short-circuits at world_size 1, and the
PyNCCL checks use a fake comm so no CUDA/NCCL is involved.
"""

import pytest
import torch


def test_destroy_distributed_restores_torch_fallback(monkeypatch):
    from freetoken.distributed.impl import DistributedCommunicator, TorchDistributedImpl, destroy_distributed

    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)

    DistributedCommunicator.plugins.append(TorchDistributedImpl())

    destroy_distributed()

    assert len(DistributedCommunicator.plugins) == 1
    assert isinstance(DistributedCommunicator.plugins[0], TorchDistributedImpl)
    # dispatch path must keep working after teardown (world_size 1 short-circuit)
    x = torch.ones(2, dtype=torch.float32)
    assert torch.equal(DistributedCommunicator().all_reduce(x), x)


def test_pynccl_wrapper_rejects_unsupported_dtype_with_guidance():
    from freetoken.distributed.impl import PyNCCLDistributedImpl

    impl = PyNCCLDistributedImpl(comm=object())
    x = torch.ones(2, dtype=torch.float32)

    with pytest.raises(TypeError, match="--disable-pynccl"):
        impl.all_reduce(x)
    with pytest.raises(TypeError, match="--disable-pynccl"):
        impl.all_gather(x)


def test_pynccl_wrapper_passes_supported_dtype_through():
    import freetoken.distributed.info as info_mod
    from freetoken.distributed.impl import PyNCCLDistributedImpl

    class FakeComm:
        def __init__(self):
            self.reduced = 0
            self.gathered = 0

        def all_reduce(self, x, op):
            assert op == "sum"
            self.reduced += 1

        def all_gather(self, out, x):
            self.gathered += 1

    # tp info is one-shot and other modules set it at import; force our geometry
    info_mod.reset_tp_info()
    info_mod.set_tp_info(rank=0, size=2)
    comm = FakeComm()
    impl = PyNCCLDistributedImpl(comm=comm)
    x = torch.ones(3, dtype=torch.bfloat16)

    impl.all_reduce(x)
    out = impl.all_gather(x)

    assert comm.reduced == 1
    assert comm.gathered == 1
    assert out.shape == (6,)


# ── real two-rank gloo ring (CPU) ───────────────────────────────────────────


def _gloo_worker(rank, world_size, init_file, out_dir):
    """Subprocess body: join the gloo group, reduce + gather, drop an ok marker."""
    import os

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        from freetoken.distributed.impl import TorchDistributedImpl

        impl = TorchDistributedImpl()
        # rank r contributes r+1; the sum over both ranks is 1+2 = 3
        y = impl.all_reduce(torch.full((4,), float(rank + 1)))
        assert torch.allclose(y, torch.full((4,), 3.0)), y
        z = impl.all_gather(torch.full((2,), float(rank)))
        assert torch.allclose(z, torch.tensor([0.0, 0.0, 1.0, 1.0])), z
    finally:
        dist.destroy_process_group()
    with open(os.path.join(out_dir, f"rank{rank}.ok"), "w") as f:
        f.write("ok")


def test_torch_distributed_impl_two_rank_gloo_ring(tmp_path):
    """Spawns two real processes so the collective code path (not the world_size-1
    short-circuit) runs end-to-end on the CPU gloo backend."""
    import torch.distributed as dist
    import torch.multiprocessing as mp

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed gloo backend is not available")

    init_file = tmp_path / "rdzv"
    mp.start_processes(
        _gloo_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    assert (tmp_path / "rank0.ok").exists() and (tmp_path / "rank1.ok").exists()

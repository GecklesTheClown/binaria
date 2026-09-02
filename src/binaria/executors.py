"""
Running independent fits in parallel.

The parallelism here is *sweep-level* and nothing else. A sweep is a few
hundred fits that share no state -- different ``k``, different seeds, no
gradients to synchronise -- so the right tool is a process pool, not
``torch.distributed``. DDP and friends exist to shard one model that is
too large for one device, and pay all-reduce traffic every step to do it.
Applied here that would be overhead for zero benefit. The threshold at
which this reverses is when a single fit stops fitting on one GPU; memory
is dominated by the ``s x i`` logit matrix, which is megabytes at the
sizes this package targets.

This module knows nothing about SiGMoiD. It maps a picklable callable
over a sequence of picklable jobs, in order. Everything model-specific
lives in ``selection``, which depends on this module and not the reverse.

Device assignment
-----------------
Workers pin one device each and report it through ``current_device()``.
The job function asks for it rather than being told, so the same function
runs unchanged in-process (where ``current_device()`` is ``None`` and the
caller's own device setting applies) and in a worker.
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Protocol, TypeVar

Job = TypeVar("Job")
Result = TypeVar("Result")

__all__ = [
    "Executor",
    "MultiGPUExecutor",
    "SerialExecutor",
    "available_devices",
    "current_device",
]


class Executor(Protocol):
    """
    Runs a callable over jobs and returns the results **in job order**.

    Ordered output is part of the contract. Seeds are computed up front,
    so a sweep's result is independent of the order fits complete in --
    but only if results are reassembled by position rather than by
    arrival. Returning a list rather than an as-completed iterator is what
    keeps callers from depending on arrival order.
    """

    def map(self, fn: Callable[[Job], Result], jobs: Sequence[Job]) -> list[Result]:
        """Apply ``fn`` to every job, returning results in job order."""
        ...


class SerialExecutor:
    """
    Runs everything in the calling process.

    The default, and the reference the parallel executors are tested
    against: a sweep run through any other executor must produce results
    identical to this one, not merely similar.
    """

    def map(self, fn: Callable[[Job], Result], jobs: Sequence[Job]) -> list[Result]:
        return [fn(job) for job in jobs]

    def __enter__(self) -> "SerialExecutor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


# Set once per worker process by the pool initializer. A module global is
# the only channel available: ProcessPoolExecutor gives no per-worker
# state, and passing the device with every job would let two jobs land on
# one device while another sits idle.
_WORKER_DEVICE: str | None = None


def current_device() -> str | None:
    """
    The device this process was pinned to, or ``None`` if it was not.

    ``None`` in the main process is meaningful rather than an error: it
    means "nobody pinned you, use whatever you were configured with".
    """
    return _WORKER_DEVICE


def available_devices() -> list[str]:
    """Every CUDA device present, or ``["cpu"]`` if there are none."""
    import torch

    if torch.cuda.is_available():
        return [f"cuda:{index}" for index in range(torch.cuda.device_count())]
    return ["cpu"]


def _pin_device(device_queue: object, threads_per_worker: int | None) -> None:
    """
    Pool initializer: claim one device for this worker, permanently.

    Claiming from a queue rather than deriving from a worker index because
    ProcessPoolExecutor does not promise stable or contiguous worker
    identities. The timeout matters: a replaced worker would find the
    queue empty, and without it the pool would hang rather than fail.

    Thread capping is not incidental. Each worker is a *separate process*,
    and torch sizes its intra-op thread pool from the machine's core count
    by default -- so N workers on a 64-core node ask for 64 threads each
    and contend for the same cores. On a shared scheduler that is worse
    than wasteful: the core count torch sees is the *node's*, not the
    cgroup's, so the oversubscription is invisible from inside the job.
    """
    global _WORKER_DEVICE
    _WORKER_DEVICE = device_queue.get(timeout=30)  # type: ignore[attr-defined]

    import torch

    if threads_per_worker is not None:
        torch.set_num_threads(threads_per_worker)

    if _WORKER_DEVICE is not None and _WORKER_DEVICE.startswith("cuda"):
        # Makes this the default device for the process, so any tensor
        # created without an explicit device still lands on the right card.
        torch.cuda.set_device(_WORKER_DEVICE)


class MultiGPUExecutor:
    """
    One worker process per device, with a persistent pool.

    Parameters
    ----------
    devices : sequence of str or None, default=None
        Devices to pin workers to, one worker each. ``None`` uses every
        CUDA device present, falling back to a single CPU worker.
    max_workers : int or None, default=None
        Cap on workers, applied after ``devices`` is resolved. Useful for
        testing the parallel path on a machine with one card.
    threads_per_worker : int or None, default=1
        Intra-op torch threads per worker process. ``None`` leaves torch's
        default alone.

        1 by default because workers are separate processes and torch
        sizes its thread pool from the machine's core count, so N workers
        on a 64-core node each ask for 64 threads. On a shared scheduler
        the count torch sees is the *node's* rather than the job's cgroup,
        so the oversubscription is invisible from inside the job. With one
        fit per GPU there is little intra-op CPU work to parallelise
        anyway. Raise it if running CPU-only workers on a node you have to
        yourself.

    Notes
    -----
    The start method is ``spawn``. CUDA and ``fork`` do not coexist: a
    forked child inherits a CUDA context it does not own and fails on
    first use, usually with an error pointing nowhere near the cause.
    ``spawn`` costs a fresh interpreter and a torch re-import per worker,
    so the pool is created once and reused across calls to ``map``.

    Jobs are dispatched one at a time (``chunksize=1``) because fits are
    not equal-cost: a large ``k`` costs several times a small one, and a
    fit that converges early costs a fraction of one that runs to
    ``max_iter``. Fixed chunks would leave workers idle at the tail.

    Running several workers per device does not help. Without MPS, CUDA
    processes time-slice rather than share. Small matrices do underfill a
    GPU, but the fix for that is batching inside one process.
    """

    def __init__(
        self,
        devices: Sequence[str] | None = None,
        *,
        max_workers: int | None = None,
        threads_per_worker: int | None = 1,
    ) -> None:
        resolved = list(devices) if devices is not None else available_devices()
        if not resolved:
            raise ValueError("MultiGPUExecutor needs at least one device")
        if max_workers is not None:
            if max_workers < 1:
                raise ValueError(f"max_workers must be >= 1, got {max_workers}")
            resolved = resolved[:max_workers]
        self.devices = resolved
        self.threads_per_worker = threads_per_worker
        self._pool: ProcessPoolExecutor | None = None

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            context = get_context("spawn")
            queue = context.Queue()
            for device in self.devices:
                queue.put(device)
            self._pool = ProcessPoolExecutor(
                max_workers=len(self.devices),
                mp_context=context,
                initializer=_pin_device,
                initargs=(queue, self.threads_per_worker),
            )
        return self._pool

    def map(self, fn: Callable[[Job], Result], jobs: Sequence[Job]) -> list[Result]:
        if not jobs:
            return []
        return list(self._ensure_pool().map(fn, jobs, chunksize=1))

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> "MultiGPUExecutor":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(devices={self.devices!r})"

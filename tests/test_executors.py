import numpy as np
import pytest
import torch

from binaria.executors import (
    MultiGPUExecutor,
    SerialExecutor,
    available_devices,
    current_device,
)
from binaria.selection import SiGMoiDSelector


def _synthetic_binary(n_samples: int, n_features: int, true_k: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    beta = torch.randn(n_samples, true_k, dtype=torch.float64)
    energy = torch.randn(true_k, n_features, dtype=torch.float64)
    return torch.bernoulli(torch.sigmoid(-(beta @ energy)))


def _double(value: int) -> int:
    # Module level so it survives pickling to a worker.
    return value * 2


# --- the protocol ---------------------------------------------------------


def test_serial_executor_preserves_job_order() -> None:
    assert SerialExecutor().map(_double, [3, 1, 2]) == [6, 2, 4]


def test_executors_return_an_empty_list_for_no_jobs() -> None:
    assert SerialExecutor().map(_double, []) == []
    with MultiGPUExecutor(devices=["cpu"]) as executor:
        assert executor.map(_double, []) == []


def test_available_devices_is_never_empty() -> None:
    assert available_devices()


def test_current_device_is_none_in_the_main_process() -> None:
    # Not an oversight: "nobody pinned you" is how the job function knows
    # to fall back to the selector's own device setting.
    assert current_device() is None


def test_multigpu_executor_rejects_a_degenerate_configuration() -> None:
    with pytest.raises(ValueError, match="at least one device"):
        MultiGPUExecutor(devices=[])
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        MultiGPUExecutor(devices=["cpu"], max_workers=0)


def test_max_workers_caps_the_device_list() -> None:
    executor = MultiGPUExecutor(devices=["cpu", "cpu", "cpu"], max_workers=2)
    assert executor.devices == ["cpu", "cpu"]


# --- the parallel path ----------------------------------------------------


def test_multigpu_executor_runs_jobs_in_worker_processes() -> None:
    with MultiGPUExecutor(devices=["cpu", "cpu"]) as executor:
        assert executor.map(_double, list(range(20))) == [v * 2 for v in range(20)]


def test_pool_is_reused_across_map_calls() -> None:
    with MultiGPUExecutor(devices=["cpu"]) as executor:
        executor.map(_double, [1])
        first = executor._pool
        executor.map(_double, [2])
        assert executor._pool is first
    assert executor._pool is None  # and shutdown releases it


# --- the property the whole design rests on -------------------------------


def test_parallel_sweep_is_bit_identical_to_serial() -> None:
    data = _synthetic_binary(40, 24, true_k=2).numpy()

    def run(executor: object) -> SiGMoiDSelector:
        selector = SiGMoiDSelector(
            [2, 3, 4],
            alpha_range=(0.0, 0.5),
            n_repeats=3,
            max_iter=40,
            random_state=0,
            executor=executor,  # type: ignore[arg-type]
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return selector.fit(data)

    serial = run(SerialExecutor())
    with MultiGPUExecutor(devices=["cpu", "cpu"]) as pool:
        parallel = run(pool)

    for column in serial.cv_results_:
        left, right = serial.cv_results_[column], parallel.cv_results_[column]
        if column == "fit_time":
            assert bool((left > 0.0).all())
            assert bool((right > 0.0).all())
            continue
        if left.dtype.kind == "f":
            # Exact, not approximate: same seeds, same order, same maths.
            assert np.array_equal(left, right), column
        else:
            assert list(left) == list(right), column

    assert serial.best_n_components_ == parallel.best_n_components_
    assert serial.best_alpha_ == parallel.best_alpha_
    assert serial.selection_rule_ == parallel.selection_rule_
    assert serial.n_repeats_ == parallel.n_repeats_


def test_default_executor_is_serial_and_leaves_no_staging_behind() -> None:
    # The serial path must not write the matrix to disk at all -- staging
    # exists only to keep jobs small when they cross a process boundary.
    import tempfile
    from pathlib import Path

    before = {p.name for p in Path(tempfile.gettempdir()).glob("binaria-sweep-*")}
    selector = SiGMoiDSelector([2, 3], alpha_range=(0.0,), n_repeats=2, max_iter=20, random_state=0)
    assert selector.executor is None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        selector.fit(_synthetic_binary(30, 16, true_k=2).numpy())

    after = {p.name for p in Path(tempfile.gettempdir()).glob("binaria-sweep-*")}
    assert after == before

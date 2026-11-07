"""Running independent simulations across worker processes."""

import multiprocessing
import os
import warnings
from collections.abc import Callable, Sequence
from concurrent import futures
from typing import Any

from msoaxon import _common, _fast

# the warnings msoaxon issues, which a pool worker may issue in the parent's
# place
_OURS = (_common.UnstableRestWarning, _fast.DriftWarning)

# every module whose functions run in pool workers
_PRELOAD = [
    "msoaxon.coincidence",
    "msoaxon.somatic",
    "msoaxon.threshold",
    "msoaxon.measure",
]


def pool_context(
    method: str | None = None,
) -> multiprocessing.context.BaseContext:
    """Returns the multiprocessing context for worker pools.

    That is the platform default, except that fork is replaced by
    forkserver. Forking a process that runs threads (a pytest-xdist
    worker, a numerical library's thread pool) can deadlock the child,
    and Python 3.12+ warns when it happens; Python 3.14 makes forkserver
    the Linux default for this reason. macOS and Windows already default
    to spawn.

    Args:
        method: The default start method to assume (for tests); None
            reads the platform's.

    Returns:
        A multiprocessing context.
    """
    method = method or multiprocessing.get_context().get_start_method()
    if method != "fork":
        return multiprocessing.get_context(method)
    ctx = multiprocessing.get_context("forkserver")
    # the server imports these once, so each worker forks with msoaxon, numpy
    # and scipy already loaded instead of importing them itself (nearly fork's
    # start-up cost; the server runs no other threads, so forking it is safe).
    # No effect once the server is running.
    ctx.set_forkserver_preload(_PRELOAD)
    return ctx


def warning_filters() -> list[tuple[Any, ...]]:
    """Returns the warning filters that can match msoaxon's own warnings.

    Those whose category is a base of UnstableRestWarning or
    DriftWarning, in order. Only these go to pool workers: a filter for
    a category a script defines could not be unpickled there.

    Returns:
        Entries of warnings.filters.
    """
    return [
        f for f in warnings.filters if any(issubclass(c, f[2]) for c in _OURS)
    ]


def worker_init(
    env_mode: str | None,
    warned: tuple[bool, bool],
    stop_when_settled: bool,
    filters: Sequence[tuple[Any, ...]] = (),
) -> None:
    """Gives a new pool worker this process's module state.

    The MSOAXON_MODE and DriftWarning state (_fast.inherit),
    _dispatch.STOP_WHEN_SETTLED, which scripts/benchmark_search.py
    toggles, and the warning filters for msoaxon's warnings, so that
    warnings.simplefilter("ignore", msoaxon.UnstableRestWarning) also
    silences runs in workers: a forkserver or spawned worker would
    otherwise keep the import defaults.

    Args:
        env_mode: The parent's MSOAXON_MODE (None if unset).
        warned: Whether the parent issued the fast and the native
            DriftWarning (_fast.parent_state()).
        stop_when_settled: The parent's _dispatch.STOP_WHEN_SETTLED.
        filters: The parent's warning_filters(), which go in front of
            the worker's own.
    """
    from msoaxon import _dispatch  # noqa: PLC0415  (imports _parallel)

    _fast.inherit(env_mode, warned)
    _dispatch.STOP_WHEN_SETTLED = stop_when_settled
    for action, message, category, module, lineno in reversed(filters):
        warnings.filterwarnings(
            action,
            message.pattern if message else "",
            category,
            module.pattern if module else "",
            lineno,
        )


def process_pool(
    max_workers: int | None = None,
) -> futures.ProcessPoolExecutor:
    """Returns a ProcessPoolExecutor using pool_context().

    Use this for every worker pool. Workers adopt this process's
    MSOAXON_MODE and DriftWarning state, its STOP_WHEN_SETTLED and its
    filters for msoaxon's warnings, as they are when the pool is made
    (worker_init): a forkserver worker would otherwise see the state of
    the server's start.

    Args:
        max_workers: The number of processes (default: all CPUs).

    Returns:
        A concurrent.futures.ProcessPoolExecutor.
    """
    from msoaxon import _dispatch  # noqa: PLC0415  (imports _parallel)

    return futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=pool_context(),
        initializer=worker_init,
        initargs=(
            *_fast.parent_state(),
            _dispatch.STOP_WHEN_SETTLED,
            warning_filters(),
        ),
    )


def map_tasks(
    fn: Callable[[Any], Any],
    tasks: Sequence[Any],
    workers: int | None = None,
    executor: futures.Executor | None = None,
    chunksize: int = 1,
) -> list[Any]:
    """Computes [fn(t) for t in tasks], in parallel unless workers == 1.

    Args:
        fn: The function to apply; it must be a module-level function
            so the workers can import it.
        tasks: The arguments, one per call.
        workers: Without an executor, the size of a pool made for this
            call (default: all CPUs); 1 (or a single task) runs
            in-process.
        executor: An existing pool to run on. Scripts that make several
            parallel calls should open one process_pool() and pass it to
            each: a fresh pool per call re-imports msoaxon in every
            worker and starts with an empty pre-stimulus cache.
        chunksize: Tasks sent to a worker at a time.

    Returns:
        The results, in the order of `tasks`.
    """
    if executor is not None:
        return list(executor.map(fn, tasks, chunksize=chunksize))
    workers = min(workers or os.cpu_count() or 1, len(tasks))
    if (
        workers <= 1
    ):  # also no tasks at all, which a pool of 0 workers would reject
        return [fn(t) for t in tasks]
    with process_pool(workers) as pool:
        return list(pool.map(fn, tasks, chunksize=chunksize))

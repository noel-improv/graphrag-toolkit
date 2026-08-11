# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run-level instrumentation for extraction benchmarks.

Records three things a run currently doesn't report: how long each phase took,
how many LLM calls were retried or throttled, and how many bytes moved.

Extraction workers are separate processes started with spawn
(`indexing/utils/pipeline_utils.py`), so counters can't accumulate in parent
memory - a child gets a fresh import and its own module globals. Each process
writes its own file into `BENCHMARK_METRICS_DIR` and `summarize()` sums them.

Retries are counted off the log record tenacity emits from its `before_sleep`
hook, which fires exactly once per retry. That is the only layer we can see:
botocore retries inside `llm_cache.py` never surface as a record, so every
retry number this module produces is *observed* retries, not total retries.
Throttle counts are the more reliable saturation signal.
"""

import atexit
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

METRICS_DIR_VAR = 'BENCHMARK_METRICS_DIR'

# tenacity's before_sleep_log hook is attached to this logger by the retry
# decorator that bedrock_utils monkeypatches onto llama_index.
RETRY_LOGGER_NAME = 'graphrag_toolkit.lexical_graph.utils.bedrock_utils'

# Exception names that mean "the service asked us to slow down", as opposed to
# a transient server-side failure. Only these count toward saturation.
THROTTLE_EXCEPTIONS = ('ThrottlingException', 'TooManyRequestsException')

# Exception names the retry decorator retries on that are *not* throttles.
SERVER_ERROR_EXCEPTIONS = (
    'InternalServerException',
    'ServiceUnavailableException',
    'ModelTimeoutException',
    'ModelErrorException',
)


class RunMetrics:
    """
    Process-local counters and phase timings, flushed to a single JSON file.

    One instance per process. Counts are plain ints guarded by a lock, because
    extraction fans out to threads within each worker process.
    """

    def __init__(self, metrics_dir: str, pid: Optional[int] = None):
        self.metrics_dir = metrics_dir
        self.pid = pid if pid is not None else os.getpid()
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._phases: List[Dict[str, Any]] = []

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def counters(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def phases(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._phases)

    def add_phase(self, name: str, seconds: float) -> None:
        with self._lock:
            self._phases.append({'name': name, 'seconds': round(seconds, 3)})

    @contextmanager
    def phase(self, name: str):
        """
        Time a named span. The span is recorded even when the body raises, so a
        run that fails partway still reports where the time went.
        """
        started = time.monotonic()
        try:
            yield
        finally:
            self.add_phase(name, time.monotonic() - started)

    def as_dict(self) -> Dict[str, Any]:
        return {'pid': self.pid, 'counters': self.counters(), 'phases': self.phases()}

    def flush(self) -> Optional[str]:
        """
        Write this process's record. Returns the path, or None if nothing was
        recorded - an empty file from a process that did no work is noise.
        """
        if not self._counters and not self._phases:
            return None

        os.makedirs(self.metrics_dir, exist_ok=True)
        path = os.path.join(self.metrics_dir, f'run-metrics-{self.pid}.json')

        # Write-then-rename, so the parent never reads a half-written file if it
        # summarizes while a worker is still shutting down.
        tmp_path = f'{path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.as_dict(), f)
        os.replace(tmp_path, path)

        return path


class RetryCountingHandler(logging.Handler):
    """
    Counts tenacity retries by reading the record its `before_sleep` hook emits.

    The record's message names the exception that triggered the retry, so a
    throttle can be told apart from a server-side error without reaching into
    tenacity's internals.
    """

    def __init__(self, metrics: RunMetrics):
        super().__init__(level=logging.WARNING)
        self.metrics = metrics

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # a broken record must never break the run
            return

        if 'Retrying' not in message:
            return

        self.metrics.increment('llm_retries_observed')

        for name in THROTTLE_EXCEPTIONS:
            if name in message:
                self.metrics.increment('llm_throttles_observed')
                return

        for name in SERVER_ERROR_EXCEPTIONS:
            if name in message:
                self.metrics.increment('llm_server_errors_observed')
                return

        self.metrics.increment('llm_retries_unclassified')


_metrics: Optional[RunMetrics] = None
_handler: Optional[RetryCountingHandler] = None
_install_lock = threading.Lock()


def install(metrics_dir: Optional[str] = None) -> Optional[RunMetrics]:
    """
    Attach the retry counter and arrange for a flush at process exit.

    Idempotent, and a no-op when no metrics directory is configured, so an
    ordinary benchmark run is unaffected until it opts in.
    """
    global _metrics, _handler

    with _install_lock:
        if _metrics is not None:
            return _metrics

        metrics_dir = metrics_dir or os.environ.get(METRICS_DIR_VAR)
        if not metrics_dir:
            return None

        _metrics = RunMetrics(metrics_dir)
        _handler = RetryCountingHandler(_metrics)

        retry_logger = logging.getLogger(RETRY_LOGGER_NAME)
        retry_logger.addHandler(_handler)
        # before_sleep_log emits at WARNING; if the run raised the threshold
        # above that the records never reach a handler at all.
        if retry_logger.level > logging.WARNING or retry_logger.level == logging.NOTSET:
            retry_logger.setLevel(logging.WARNING)

        atexit.register(_metrics.flush)

        return _metrics


def get_metrics() -> Optional[RunMetrics]:
    return _metrics


def increment(name: str, amount: int = 1) -> None:
    """Increment a counter if instrumentation is installed, otherwise do nothing."""
    if _metrics is not None:
        _metrics.increment(name, amount)


@contextmanager
def phase(name: str):
    """
    Time a phase, whether or not instrumentation is installed.

    Always logs the duration, so a run without a metrics directory still leaves
    per-phase timings in the log rather than only the single end-to-end number.
    """
    started = time.monotonic()
    try:
        yield
    finally:
        seconds = time.monotonic() - started
        if _metrics is not None:
            _metrics.add_phase(name, seconds)
        logger.info(f'Phase complete [phase: {name}, seconds: {seconds:.3f}]')


def default_retry_log_paths() -> List[str]:
    """
    The files a run's retry warnings land in, whichever process emitted them.

    `init_with_test_details` points graphrag_toolkit's logging at
    `test-logs/<NN>-<Name>.log`, but that config lives in the parent only - a
    spawn-started child starts with no logging config, so its warnings go to
    stderr instead. The CFN template starts the suite under `screen -L`, which
    captures the session's stdout and stderr into `screenlog.0`, so that is
    where child retries end up.

    `BENCHMARK_RETRY_LOGS` overrides this with a comma-separated list.
    """
    override = os.environ.get('BENCHMARK_RETRY_LOGS')
    if override:
        return [path.strip() for path in override.split(',') if path.strip()]

    paths = []

    if os.path.isdir('test-logs'):
        paths.extend(
            os.path.join('test-logs', name)
            for name in sorted(os.listdir('test-logs'))
            if name.endswith('.log')
        )

    if os.path.exists('screenlog.0'):
        paths.append('screenlog.0')

    return paths


def count_retries_in_logs(log_paths: List[str]) -> Dict[str, int]:
    """
    Count retries by reading log files rather than by counting in process.

    Extraction runs its transformations in spawn-started child processes
    (`extraction_pipeline.py` -> `run_pipeline`), and that happens even at
    num_workers=1. A child re-imports from scratch, so `install()` never runs
    there and the in-process handler sees none of the extraction retries.
    Children do inherit the parent's stderr, so their warnings land in the same
    log the harness already captures - which is where these counts come from.

    Use this for extraction. The in-process handler still covers the query
    phase, which runs in the parent.
    """
    counters: Dict[str, int] = {}

    def _increment(name: str) -> None:
        counters[name] = counters.get(name, 0) + 1

    for path in log_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    if 'Retrying' not in line or 'as it raised' not in line:
                        continue

                    _increment('llm_retries_observed')

                    if any(name in line for name in THROTTLE_EXCEPTIONS):
                        _increment('llm_throttles_observed')
                    elif any(name in line for name in SERVER_ERROR_EXCEPTIONS):
                        _increment('llm_server_errors_observed')
                    else:
                        _increment('llm_retries_unclassified')
        except OSError as e:
            logger.warning(f'Skipping unreadable log file [path: {path}, error: {e}]')

    return counters


def summarize(metrics_dir: str) -> Dict[str, Any]:
    """
    Sum every process record in `metrics_dir` into one result.

    Counters add across processes. Phase timings do not: workers run
    concurrently, so their spans overlap and a sum would exceed wall time.
    Each phase is reported with its total, max and the number of processes
    that recorded it, which is what tells overlap apart from serial work.
    """
    records = []

    if os.path.isdir(metrics_dir):
        for filename in sorted(os.listdir(metrics_dir)):
            if not (filename.startswith('run-metrics-') and filename.endswith('.json')):
                continue
            path = os.path.join(metrics_dir, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    records.append(json.load(f))
            except (OSError, ValueError) as e:
                # One unreadable worker file shouldn't discard the whole run.
                logger.warning(f'Skipping unreadable metrics file [path: {path}, error: {e}]')

    counters: Dict[str, int] = {}
    phases: Dict[str, Dict[str, Any]] = {}

    for record in records:
        for name, value in record.get('counters', {}).items():
            counters[name] = counters.get(name, 0) + value

        for entry in record.get('phases', []):
            name = entry.get('name')
            seconds = entry.get('seconds', 0.0)
            if name is None:
                continue
            phase_stats = phases.setdefault(
                name, {'total_seconds': 0.0, 'max_seconds': 0.0, 'num_processes': 0}
            )
            phase_stats['total_seconds'] = round(phase_stats['total_seconds'] + seconds, 3)
            phase_stats['max_seconds'] = round(max(phase_stats['max_seconds'], seconds), 3)
            phase_stats['num_processes'] += 1

    return {'num_processes': len(records), 'counters': counters, 'phases': phases}

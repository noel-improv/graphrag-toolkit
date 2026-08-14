# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import tempfile
import unittest

from benchmarks.utils.run_metrics import (
    RetryCountingHandler,
    RunMetrics,
    count_retries_in_logs,
    summarize,
)

# The line tenacity's before_sleep_log hook writes, captured from a real run of
# the retry decorator in bedrock_utils.
RETRY_LINE = (
    'WARNING:graphrag_toolkit.lexical_graph.utils.bedrock_utils:'
    'Retrying graphrag.extract in 4 seconds as it raised {exception}: rate exceeded.'
)


class TestRunMetricsCounters(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.metrics = RunMetrics(self.tmp.name, pid=101)

    def tearDown(self):
        self.tmp.cleanup()

    def test_increment_accumulates(self):
        self.metrics.increment('llm_calls')
        self.metrics.increment('llm_calls', 4)

        self.assertEqual(self.metrics.counters()['llm_calls'], 5)

    def test_phase_records_duration(self):
        with self.metrics.phase('extract'):
            pass

        phases = self.metrics.phases()
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0]['name'], 'extract')
        self.assertGreaterEqual(phases[0]['seconds'], 0.0)

    def test_phase_records_duration_when_body_raises(self):
        """A run that fails partway still reports where the time went."""
        with self.assertRaises(ValueError):
            with self.metrics.phase('extract'):
                raise ValueError('boom')

        self.assertEqual(self.metrics.phases()[0]['name'], 'extract')

    def test_flush_writes_record(self):
        self.metrics.increment('llm_calls', 3)
        path = self.metrics.flush()

        with open(path, 'r', encoding='utf-8') as f:
            record = json.load(f)

        self.assertEqual(record['pid'], 101)
        self.assertEqual(record['counters']['llm_calls'], 3)

    def test_flush_skips_empty_record(self):
        """A worker process that did no work shouldn't leave a file behind."""
        self.assertIsNone(self.metrics.flush())
        self.assertEqual(os.listdir(self.tmp.name), [])


class TestRetryCountingHandler(unittest.TestCase):
    """
    Counts are driven by the log record tenacity's before_sleep_log hook emits,
    so the tests feed the handler records in that shape.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.metrics = RunMetrics(self.tmp.name, pid=1)
        self.handler = RetryCountingHandler(self.metrics)

    def tearDown(self):
        self.tmp.cleanup()

    def _emit(self, message):
        self.handler.emit(
            logging.LogRecord(
                name='graphrag_toolkit.lexical_graph.utils.bedrock_utils',
                level=logging.WARNING,
                pathname=__file__,
                lineno=1,
                msg=message,
                args=(),
                exc_info=None,
            )
        )

    def test_throttle_counts_as_retry_and_throttle(self):
        self._emit('Retrying converse in 4.0 seconds as it raised ThrottlingException: rate exceeded.')

        counters = self.metrics.counters()
        self.assertEqual(counters['llm_retries_observed'], 1)
        self.assertEqual(counters['llm_throttles_observed'], 1)
        self.assertNotIn('llm_server_errors_observed', counters)

    def test_server_error_is_not_counted_as_a_throttle(self):
        """Saturation is measured by throttles; a 500 is a different signal."""
        self._emit('Retrying converse in 4.0 seconds as it raised InternalServerException: oops.')

        counters = self.metrics.counters()
        self.assertEqual(counters['llm_retries_observed'], 1)
        self.assertEqual(counters['llm_server_errors_observed'], 1)
        self.assertNotIn('llm_throttles_observed', counters)

    def test_unrecognised_exception_is_counted_but_flagged(self):
        self._emit('Retrying converse in 4.0 seconds as it raised SomeNewException: hmm.')

        counters = self.metrics.counters()
        self.assertEqual(counters['llm_retries_observed'], 1)
        self.assertEqual(counters['llm_retries_unclassified'], 1)

    def test_unrelated_warning_is_ignored(self):
        self._emit('Connection pool is full, discarding connection')

        self.assertEqual(self.metrics.counters(), {})

    def test_handler_attached_to_logger_counts_real_records(self):
        """The counter has to fire through the logging system, not just directly."""
        real_logger = logging.getLogger('graphrag_toolkit.lexical_graph.utils.bedrock_utils')
        real_logger.addHandler(self.handler)
        real_logger.setLevel(logging.WARNING)
        try:
            real_logger.warning(
                'Retrying %s in %s seconds as it raised ThrottlingException: rate exceeded.',
                'converse',
                4.0,
            )
        finally:
            real_logger.removeHandler(self.handler)

        self.assertEqual(self.metrics.counters()['llm_throttles_observed'], 1)


class TestSummarize(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, metrics):
        metrics.flush()

    def test_counters_sum_across_processes(self):
        for pid, calls in ((1, 10), (2, 5)):
            metrics = RunMetrics(self.tmp.name, pid=pid)
            metrics.increment('llm_calls', calls)
            metrics.increment('llm_throttles_observed', 1)
            self._write(metrics)

        result = summarize(self.tmp.name)

        self.assertEqual(result['num_processes'], 2)
        self.assertEqual(result['counters']['llm_calls'], 15)
        self.assertEqual(result['counters']['llm_throttles_observed'], 2)

    def test_phases_report_total_and_max_separately(self):
        """
        Workers overlap, so summing their spans would exceed wall time. The max
        is what compares against it.
        """
        for pid, seconds in ((1, 10.0), (2, 4.0)):
            metrics = RunMetrics(self.tmp.name, pid=pid)
            metrics.add_phase('extract', seconds)
            self._write(metrics)

        phase = summarize(self.tmp.name)['phases']['extract']

        self.assertEqual(phase['total_seconds'], 14.0)
        self.assertEqual(phase['max_seconds'], 10.0)
        self.assertEqual(phase['num_processes'], 2)

    def test_missing_directory_returns_empty_summary(self):
        result = summarize(os.path.join(self.tmp.name, 'does-not-exist'))

        self.assertEqual(result['num_processes'], 0)
        self.assertEqual(result['counters'], {})

    def test_unreadable_file_does_not_discard_the_run(self):
        metrics = RunMetrics(self.tmp.name, pid=1)
        metrics.increment('llm_calls', 7)
        self._write(metrics)

        with open(os.path.join(self.tmp.name, 'run-metrics-999.json'), 'w', encoding='utf-8') as f:
            f.write('{ truncated')

        result = summarize(self.tmp.name)

        self.assertEqual(result['counters']['llm_calls'], 7)


class TestCountRetriesInLogs(unittest.TestCase):
    """
    Extraction runs in spawn-started children, so its retries only ever appear
    in the log. These cover the path that actually counts extraction retries.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_log(self, name, lines):
        path = os.path.join(self.tmp.name, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        return path

    def test_counts_throttles_and_server_errors_separately(self):
        path = self._write_log('extract.log', [
            RETRY_LINE.format(exception='ThrottlingException'),
            RETRY_LINE.format(exception='ThrottlingException'),
            RETRY_LINE.format(exception='ModelTimeoutException'),
            'INFO:some.other.module:Running extraction pipeline [batch: 1]',
        ])

        counters = count_retries_in_logs([path])

        self.assertEqual(counters['llm_retries_observed'], 3)
        self.assertEqual(counters['llm_throttles_observed'], 2)
        self.assertEqual(counters['llm_server_errors_observed'], 1)

    def test_counts_across_multiple_log_files(self):
        paths = [
            self._write_log('a.log', [RETRY_LINE.format(exception='ThrottlingException')]),
            self._write_log('b.log', [RETRY_LINE.format(exception='ThrottlingException')]),
        ]

        self.assertEqual(count_retries_in_logs(paths)['llm_throttles_observed'], 2)

    def test_log_without_retries_returns_empty(self):
        path = self._write_log('quiet.log', ['INFO:x:nothing to see'])

        self.assertEqual(count_retries_in_logs([path]), {})

    def test_missing_log_file_does_not_raise(self):
        path = os.path.join(self.tmp.name, 'does-not-exist.log')

        self.assertEqual(count_retries_in_logs([path]), {})


if __name__ == '__main__':
    unittest.main()

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

from graphrag_toolkit.lexical_graph import GraphRAGConfig


SWEEP_ENV = {
    'AWS_REGION_NAME': 'us-west-2',
    'AWS_DEFAULT_REGION': 'us-west-2',
    'S3_RESULTS_BUCKET': 'test-bucket',
    'S3_RESULTS_PREFIX': 'test-prefix',
    'GRAPH_STORE': 'dummy://graph',
    'VECTOR_STORE': 'dummy://vector',
    'EXTRACTION_NUM_WORKERS': '1',
    'EXTRACTION_THREAD_SWEEP': '64,8,1',
}


def _context_manager(value):
    manager = MagicMock()
    manager.__enter__ = MagicMock(return_value=value)
    manager.__exit__ = MagicMock(return_value=False)
    return manager


class TestThreadSweepAppliesThreadCount(unittest.TestCase):
    """
    The sweep only means anything if each setting actually reaches the
    extractors.

    LexicalGraphIndex.__init__ configures the extraction pipeline and stores the
    components, and TopicExtractor reads extraction_num_threads_per_worker at
    construction. An index built once and reused would run every setting at the
    count current when it was built, and the sweep would still pass while
    reporting a flat curve. These tests fail if that regresses.
    """

    def setUp(self):
        self.handler = MagicMock()
        self.handler.add_output = MagicMock()
        self.handler.run_assertions = MagicMock()

        # Thread count observed at each LexicalGraphIndex construction.
        self.observed_thread_counts = []

    def _run(self):
        from benchmarks.scripts import benchmark_extract

        def _capture(*args, **kwargs):
            self.observed_thread_counts.append(
                GraphRAGConfig.extraction_num_threads_per_worker
            )
            return MagicMock()

        with (
            patch.dict('os.environ', SWEEP_ENV, clear=False),
            patch.object(benchmark_extract, 'sync_benchmark_data_from_s3'),
            patch.object(benchmark_extract, 'LexicalGraphIndex', side_effect=_capture) as index_cls,
            patch.object(benchmark_extract, 'S3BasedDocs'),
            patch.object(benchmark_extract, 'SimpleDirectoryReader') as reader,
            patch.object(benchmark_extract, '_count_source_docs', return_value=2),
            patch.object(benchmark_extract.GraphStoreFactory, 'for_graph_store',
                         return_value=_context_manager(MagicMock())),
            patch.object(benchmark_extract.VectorStoreFactory, 'for_vector_store',
                         return_value=_context_manager(MagicMock())),
        ):
            reader.return_value.load_data.return_value = [MagicMock(), MagicMock()]

            benchmark_extract.run_thread_sweep(
                self.handler, 'wikihow-subset', 'source-data', expected_docs=2
            )

        return index_cls

    def test_index_is_rebuilt_for_every_setting(self):
        index_cls = self._run()

        self.assertEqual(index_cls.call_count, 3)

    def test_each_setting_is_applied_before_the_index_is_built(self):
        """The regression that made a real sweep run every point at one count."""
        self._run()

        self.assertEqual(self.observed_thread_counts, [64, 8, 1])

    def test_results_recorded_per_setting(self):
        self._run()

        recorded = [
            call.args[0] for call in self.handler.add_output.call_args_list
            if isinstance(call.args[0], str) and call.args[0].startswith('sweep_')
        ]

        self.assertEqual(recorded, ['sweep_64_threads', 'sweep_8_threads', 'sweep_1_threads'])


class TestThreadSweepFailureHandling(unittest.TestCase):

    def setUp(self):
        self.handler = MagicMock()

    def test_one_failing_setting_does_not_lose_the_others(self):
        from benchmarks.scripts import benchmark_extract

        calls = {'n': 0}

        def _index(*args, **kwargs):
            index = MagicMock()

            def _extract(*a, **kw):
                calls['n'] += 1
                if calls['n'] == 2:
                    raise RuntimeError('throttled out')

            index.extract = _extract
            return index

        with (
            patch.dict('os.environ', SWEEP_ENV, clear=False),
            patch.object(benchmark_extract, 'sync_benchmark_data_from_s3'),
            patch.object(benchmark_extract, 'LexicalGraphIndex', side_effect=_index),
            patch.object(benchmark_extract, 'S3BasedDocs'),
            patch.object(benchmark_extract, 'SimpleDirectoryReader') as reader,
            patch.object(benchmark_extract, '_count_source_docs', return_value=2),
            patch.object(benchmark_extract.GraphStoreFactory, 'for_graph_store',
                         return_value=_context_manager(MagicMock())),
            patch.object(benchmark_extract.VectorStoreFactory, 'for_vector_store',
                         return_value=_context_manager(MagicMock())),
        ):
            reader.return_value.load_data.return_value = [MagicMock(), MagicMock()]

            benchmark_extract.run_thread_sweep(
                self.handler, 'wikihow-subset', 'source-data', expected_docs=2
            )

        sweep = [
            call.args[1] for call in self.handler.add_output.call_args_list
            if call.args[0] == 'thread_sweep'
        ][0]

        self.assertEqual(len(sweep), 3)
        self.assertIsNone(sweep[0]['error'])
        self.assertIn('throttled out', sweep[1]['error'])
        self.assertIsNone(sweep[2]['error'])


if __name__ == '__main__':
    unittest.main()

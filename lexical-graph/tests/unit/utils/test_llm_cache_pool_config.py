# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from graphrag_toolkit.lexical_graph.config import (
    GraphRAGConfig,
    BOTOCORE_DEFAULT_MAX_POOL_CONNECTIONS,
)
from graphrag_toolkit.lexical_graph.utils.llm_cache import (
    MAX_ATTEMPTS,
    TIMEOUT,
    bedrock_client_config,
)


class TestBedrockClientPoolSizing(unittest.TestCase):
    """
    Extraction runs one concurrent Bedrock request per thread. Without a pool
    size the client took botocore's default of 10, so past ten threads botocore
    discarded and reopened connections and the extra threads bought nothing.

    A 500-document sweep measured achieved concurrency flat at ~16 in-flight
    requests across 32, 64 and 128 threads while Bedrock's own latency stayed
    at ~6.6s, which is what these tests exist to stop regressing.
    """

    def setUp(self):
        self._original = GraphRAGConfig.extraction_num_threads_per_worker

    def tearDown(self):
        GraphRAGConfig.extraction_num_threads_per_worker = self._original

    def test_pool_is_sized_to_the_thread_count(self):
        GraphRAGConfig.extraction_num_threads_per_worker = 32

        self.assertEqual(bedrock_client_config().max_pool_connections, 32)

    def test_pool_never_drops_below_the_botocore_default(self):
        """A low thread count must not shrink the pool below stock behaviour."""
        GraphRAGConfig.extraction_num_threads_per_worker = 2

        self.assertEqual(
            bedrock_client_config().max_pool_connections,
            BOTOCORE_DEFAULT_MAX_POOL_CONNECTIONS,
        )

    def test_pool_tracks_high_thread_counts(self):
        GraphRAGConfig.extraction_num_threads_per_worker = 128

        self.assertEqual(bedrock_client_config().max_pool_connections, 128)

    def test_pool_is_never_smaller_than_the_threads_using_it(self):
        """The defect was a pool smaller than the thread count, at any setting."""
        for num_threads in (1, 4, 8, 10, 16, 32, 64, 128):
            with self.subTest(num_threads=num_threads):
                GraphRAGConfig.extraction_num_threads_per_worker = num_threads

                self.assertGreaterEqual(
                    bedrock_client_config().max_pool_connections, num_threads
                )

    def test_unusable_thread_count_falls_back_instead_of_raising(self):
        """
        Pool sizing sits inside client creation, so a bad value must not stop a
        client being built - it would surface as an unrelated ModelError.
        """
        for value in (None, 'lots', 0, -1):
            with self.subTest(value=value):
                GraphRAGConfig._extraction_num_threads_per_worker = value

                self.assertEqual(
                    bedrock_client_config().max_pool_connections,
                    BOTOCORE_DEFAULT_MAX_POOL_CONNECTIONS,
                )

    def test_retry_and_timeout_settings_are_unchanged(self):
        """Pool sizing must not disturb the retry or timeout behaviour."""
        GraphRAGConfig.extraction_num_threads_per_worker = 16
        config = bedrock_client_config()

        self.assertEqual(config.retries['max_attempts'], MAX_ATTEMPTS)
        self.assertEqual(config.retries['mode'], 'standard')
        self.assertEqual(config.connect_timeout, TIMEOUT)
        self.assertEqual(config.read_timeout, TIMEOUT)


if __name__ == '__main__':
    unittest.main()

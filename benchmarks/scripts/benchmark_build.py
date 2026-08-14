# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from contextlib import nullcontext
from typing import Dict, Any, Optional
import logging

from benchmarks.scripts.integration_test_base import IntegrationTestBase
from benchmarks.scripts.integration_test_handler import IntegrationTestHandler
from benchmarks.utils.s3_utils import sync_benchmark_data_from_s3

from graphrag_toolkit.lexical_graph import LexicalGraphIndex
from graphrag_toolkit.lexical_graph import GraphRAGConfig
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory
from graphrag_toolkit.lexical_graph.storage.graph import NonRedactedGraphQueryLogFormatting
from graphrag_toolkit.lexical_graph.indexing.load import FileBasedDocs, S3BasedDocs

logger = logging.getLogger(__name__)


def _latest_s3_collection(bucket_name: str, key_prefix: str) -> str:
    """
    Pick the most recent collection under a doc-store prefix.

    Extract stamps a new timestamped collection every run, so build has no way
    to name one in advance. Collection ids are `YYYYMMDD-HHMMSS`, which sorts
    lexicographically in time order, so the last prefix is the newest.

    Set BENCHMARK_COLLECTION_ID to read a specific one instead - needed whenever
    a prefix holds more than one collection, as a thread sweep leaves behind.
    """
    paginator = GraphRAGConfig.s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(
        Bucket=bucket_name, Prefix=f'{key_prefix}/', Delimiter='/'
    )

    collections = sorted(
        prefix['Prefix'].rstrip('/').rsplit('/', 1)[-1]
        for page in pages
        for prefix in page.get('CommonPrefixes', [])
    )

    if not collections:
        raise ValueError(
            f'No collections found under s3://{bucket_name}/{key_prefix}/. '
            'Check that extraction ran with BENCHMARK_DOC_STORE=s3.'
        )

    if len(collections) > 1:
        logger.warning(
            f'Multiple collections found, using the most recent '
            f'[collections: {collections}, using: {collections[-1]}]. '
            'Set BENCHMARK_COLLECTION_ID to choose explicitly.'
        )

    return collections[-1]


def _create_doc_store(dataset: str, data_dir: str, config: Dict[str, Any]):
    """
    Build the doc store extraction wrote to.

    BENCHMARK_DOC_STORE selects the backend and has to match what extract used;
    build reading local disk after extract wrote to S3 is the mismatch this
    exists to remove.
    """
    doc_store = os.environ.get('BENCHMARK_DOC_STORE', 'file').lower()

    if doc_store != 's3':
        return FileBasedDocs(
            docs_directory=os.path.join(
                data_dir, dataset, config.get('extracted_dir', 'extracted')
            ),
            collection_id=config.get('collection_id', dataset)
        )

    missing = sorted(
        var for var in ('AWS_REGION_NAME', 'S3_RESULTS_BUCKET', 'S3_RESULTS_PREFIX')
        if not os.environ.get(var)
    )
    if missing:
        raise ValueError(
            f'BENCHMARK_DOC_STORE=s3 requires {", ".join(missing)} to be set'
        )

    # Region before any S3 call, so the client is not built against the ambient one.
    GraphRAGConfig.aws_region = os.environ['AWS_REGION_NAME']

    bucket_name = os.environ['S3_RESULTS_BUCKET']
    key_prefix = f'{os.environ["S3_RESULTS_PREFIX"]}/doc-store/{dataset}'

    collection_id = os.environ.get('BENCHMARK_COLLECTION_ID') or _latest_s3_collection(
        bucket_name, key_prefix
    )

    logger.info(
        f'Reading extracted docs from S3 '
        f'[bucket: {bucket_name}, prefix: {key_prefix}, collection_id: {collection_id}]'
    )

    return S3BasedDocs(
        region=os.environ['AWS_REGION_NAME'],
        bucket_name=bucket_name,
        key_prefix=key_prefix,
        collection_id=collection_id,
        for_jsonl=os.environ.get('BENCHMARK_S3_JSONL', 'false').lower() == 'true'
    )


DATASET_CONFIG = {
    'cuad-prototype': {
        'num_docs': 2,
        'extracted_dir': os.path.join('extracted', '2026-04-16'),
    },
    'cuad': {
        'num_docs': 510,
        'extracted_dir': os.path.join('extracted', '2026-02-17'),
    },
    'pga': {
        'num_docs': 507,
    },
    'concurrentqa': {
        'num_docs': 13501,
        'extracted_dir': 'extracted',
        'collection_id': '20260513-174224',
    },
    'concurrentqa-prototype': {
        'num_docs': 2,
        'extracted_dir': 'extracted',
    },
    'wikihow': {
        'num_docs': 5000,
    },
}

BENCHMARK_DATA_DIR = 'source-data'


def run_benchmark_build(handler: IntegrationTestHandler, 
                        dataset: str, 
                        data_dir: str,
                        graph_store_conn: Optional[str] = None, vector_store_conn: Optional[str] = None):
    """
    Builds graph and vector stores from pre-extracted document chunks for a benchmark dataset.

    Loads extracted chunks via FileBasedDocs, builds the graph and vector indexes, and
    asserts that the expected number of source nodes were created. Either store connection
    can be omitted — the build will proceed with whichever stores are provided, and the
    source node assertion will be skipped if no graph store is configured.

    Args:
        handler: Integration test handler for recording assertions and output.
        dataset: Dataset key (e.g. 'cuad', 'pga', 'concurrentqa'). Must have a
            corresponding entry in DATASET_CONFIG.
        data_dir: Root path to the benchmark data directory containing dataset subdirectories.
        graph_store_conn: Optional graph store connection string (e.g. 'neptune-db://<hostname>' or 
            'neptune-graph://<graph-id>').
        vector_store_conn: Optional vector store connection string (e.g. 'aoss://...').
    """
    sync_benchmark_data_from_s3(dataset, data_dir)

    config = DATASET_CONFIG.get(dataset, {})

    GraphRAGConfig.build_num_workers = 2
    GraphRAGConfig.build_batch_size = 25
    GraphRAGConfig.build_batch_write_size = 50

    docs = _create_doc_store(dataset, data_dir, config)

    handler.add_output('doc_store', type(docs).__name__)
    handler.add_output('collection_id', docs.collection_id)

    graph_ctx = GraphStoreFactory.for_graph_store(
        graph_store_conn, log_formatting=NonRedactedGraphQueryLogFormatting()
    ) if graph_store_conn else nullcontext()

    vector_ctx = VectorStoreFactory.for_vector_store(
        vector_store_conn
    ) if vector_store_conn else nullcontext()

    with graph_ctx as graph_store, vector_ctx as vector_store:
        graph_index = LexicalGraphIndex(graph_store, vector_store)
        graph_index.build(docs, show_progress=True)

        expected_num_docs = config.get('num_docs')

        class BenchmarkBuildAssertions(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                cls._graph_store = graph_store
                cls._expected_num_docs = expected_num_docs

            def test_one_source_node_for_each_doc(self):
                """Graph contains one source node per document"""
                if self._graph_store is None:
                    self.skipTest('No graph store configured')
                results = self._graph_store.execute_query('MATCH (n:`__Source__`) RETURN count(n) AS count')
                source_node_count = results[0]['count']
                if self._expected_num_docs is not None:
                    self.assertEqual(source_node_count, self._expected_num_docs)
                else:
                    self.assertGreater(source_node_count, 0)

        handler.run_assertions(BenchmarkBuildAssertions)


class CuadBenchmarkBuild(IntegrationTestBase):

    @property
    def description(self):
        return 'Build graph and vector stores from CUAD pre-extracted chunks for benchmarking'

    def _run_test(self, handler: IntegrationTestHandler, params: Dict[str, Any]):
        graph_store_conn = os.environ.get('GRAPH_STORE')
        vector_store_conn = os.environ.get('VECTOR_STORE')
        is_prototype = os.environ.get('BENCHMARK_IS_PROTOTYPE')
        if is_prototype == 'true':
            dataset_name = 'cuad-prototype' 
        else:
            dataset_name = 'cuad'

        run_benchmark_build(handler, dataset_name, BENCHMARK_DATA_DIR, graph_store_conn, vector_store_conn)


class ConcurrentQaBenchmarkBuild(IntegrationTestBase):

    @property
    def description(self):
        return 'Build graph and vector stores from ConcurrentQA pre-extracted chunks for benchmarking'

    def _run_test(self, handler: IntegrationTestHandler, params: Dict[str, Any]):
        graph_store_conn = os.environ.get('GRAPH_STORE')
        vector_store_conn = os.environ.get('VECTOR_STORE')
        is_prototype = os.environ.get('BENCHMARK_IS_PROTOTYPE')
        if is_prototype == 'true':
            dataset_name = 'concurrentqa-prototype'
        else:
            dataset_name = 'concurrentqa'

        run_benchmark_build(handler, dataset_name, BENCHMARK_DATA_DIR, graph_store_conn, vector_store_conn)


class WikihowBenchmarkBuild(IntegrationTestBase):

    @property
    def description(self):
        return 'Build graph and vector stores from WikiHow pre-extracted chunks for benchmarking'

    def _run_test(self, handler: IntegrationTestHandler, params: Dict[str, Any]):
        graph_store_conn = os.environ.get('GRAPH_STORE')
        vector_store_conn = os.environ.get('VECTOR_STORE')

        run_benchmark_build(handler, 'wikihow', BENCHMARK_DATA_DIR, graph_store_conn, vector_store_conn)


class PgaBenchmarkBuild(IntegrationTestBase):

    @property
    def description(self):
        return 'Build graph and vector stores from PGA pre-extracted chunks for benchmarking'

    def _run_test(self, handler: IntegrationTestHandler, params: Dict[str, Any]):
        graph_store_conn = os.environ.get('GRAPH_STORE')
        vector_store_conn = os.environ.get('VECTOR_STORE')

        run_benchmark_build(handler, 'pga', BENCHMARK_DATA_DIR, graph_store_conn, vector_store_conn)

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import logging
import unittest
from typing import Dict, Any, Optional

from benchmarks.scripts.integration_test_base import IntegrationTestBase
from benchmarks.scripts.integration_test_handler import IntegrationTestHandler
from benchmarks.utils.s3_utils import sync_benchmark_data_from_s3
from benchmarks.utils import run_metrics

from graphrag_toolkit.lexical_graph import LexicalGraphIndex
from graphrag_toolkit.lexical_graph import GraphRAGConfig, IndexingConfig
from graphrag_toolkit.lexical_graph.storage import GraphStoreFactory
from graphrag_toolkit.lexical_graph.storage import VectorStoreFactory
from graphrag_toolkit.lexical_graph.storage.graph import NonRedactedGraphQueryLogFormatting
from graphrag_toolkit.lexical_graph.indexing.load import FileBasedDocs, S3BasedDocs
from graphrag_toolkit.lexical_graph.indexing.extract import BatchConfig

from llama_index.core import SimpleDirectoryReader

logger = logging.getLogger(__name__)

BENCHMARK_DATA_DIR = 'source-data'


def _count_source_docs(extracted_docs) -> int:
    """
    Count extracted source documents without pulling their contents back.

    Iterating S3BasedDocs downloads every object it lists - ~16.5k GETs on the
    WikiHow run purely to produce a number. Source documents are one key prefix
    each in both layouts, so a delimiter listing counts them without bodies.
    """
    if not isinstance(extracted_docs, S3BasedDocs):
        return sum(1 for _ in extracted_docs)

    collection_path = os.path.join(
        extracted_docs.key_prefix, extracted_docs.collection_id, ''
    )
    pages = GraphRAGConfig.s3.get_paginator('list_objects_v2').paginate(
        Bucket=extracted_docs.bucket_name, Prefix=collection_path, Delimiter='/'
    )

    return sum(len(page.get('CommonPrefixes', [])) for page in pages)


def run_benchmark_extract(handler: IntegrationTestHandler,
                          dataset_name: str,
                          data_dir: str,
                          expected_docs: int,
                          use_batch: bool = True):
    """
    Extracts propositions and topics from benchmark dataset documents.

    Reads raw documents from <data_dir>/<dataset_name>/documents/, runs LLM-based
    extraction (propositions + topics), and writes results to
    <data_dir>/<dataset_name>/extracted/.

    When use_batch=True, uses Bedrock batch inference for faster extraction on
    large datasets. Requires BATCH_INFERENCE_ROLE, S3_RESULTS_BUCKET,
    S3_RESULTS_PREFIX, and AWS_REGION_NAME environment variables.

    Args:
        handler: Integration test handler for recording assertions and output.
        dataset_name: Dataset key (e.g. 'concurrentqa', 'wikihow', 'pga').
        data_dir: Root path to the benchmark data directory.
        expected_docs: Expected number of source documents (for assertion).
        use_batch: Whether to use Bedrock batch inference (default: True).
    """
    input_path = os.path.join(data_dir, dataset_name, 'documents')

    # No-op unless BENCHMARK_METRICS_DIR is set, so an ordinary run is unchanged.
    run_metrics.install()

    with run_metrics.phase('sync_benchmark_data'):
        sync_benchmark_data_from_s3(dataset_name, data_dir)

    doc_store = os.environ.get('BENCHMARK_DOC_STORE', 'file').lower()

    # Check both paths' variables before anything dereferences os.environ with
    # bracket access, or the batch block's bare KeyError fires first.
    required_vars = set()
    if use_batch:
        required_vars.update(
            ('AWS_REGION_NAME', 'S3_RESULTS_BUCKET', 'S3_RESULTS_PREFIX', 'BATCH_INFERENCE_ROLE')
        )
    if doc_store == 's3':
        required_vars.update(('AWS_REGION_NAME', 'S3_RESULTS_BUCKET', 'S3_RESULTS_PREFIX'))

    missing = sorted(var for var in required_vars if not os.environ.get(var))
    if missing:
        raise ValueError(
            f'use_batch={use_batch}, BENCHMARK_DOC_STORE={doc_store} '
            f'requires {", ".join(missing)} to be set'
        )

    GraphRAGConfig.extraction_llm = os.environ.get(
        'TEST_EXTRACTION_LLM', 'us.anthropic.claude-sonnet-4-6'
    )
    GraphRAGConfig.extraction_batch_size = 15000

    # Both knobs are overridable so a sweep can vary one per run. The defaults
    # are the values this harness has always used, so an unconfigured run is
    # unchanged. A single-worker baseline needs EXTRACTION_NUM_WORKERS=1.
    GraphRAGConfig.extraction_num_workers = int(os.environ.get('EXTRACTION_NUM_WORKERS', 2))

    num_threads = os.environ.get('EXTRACTION_NUM_THREADS_PER_WORKER')
    if num_threads:
        GraphRAGConfig.extraction_num_threads_per_worker = int(num_threads)

    # This knob drives LLM concurrency only on the on-demand path
    # (TopicExtractor / LLMPropositionExtractor take it as num_workers). Under
    # batch inference the Batch* extractors submit jobs instead, and it reaches
    # nothing but the S3 read and write pools - so a thread sweep run with
    # use_batch=True measures S3 IO, not extraction concurrency.
    logger.info(
        f'Extraction concurrency [num_workers: {GraphRAGConfig.extraction_num_workers}, '
        f'num_threads_per_worker: {GraphRAGConfig.extraction_num_threads_per_worker}, '
        f'use_batch: {use_batch}]'
    )

    # The harness exports AWS_REGION_NAME, but boto3 clients take their region
    # from GraphRAGConfig. The region= argument on BatchConfig and S3BasedDocs is
    # stored and never read, so without this they can target the wrong region.
    aws_region = os.environ.get('AWS_REGION_NAME')
    if aws_region:
        GraphRAGConfig.aws_region = aws_region

    indexing_config = None
    if use_batch:
        batch_config = BatchConfig(
            region=os.environ['AWS_REGION_NAME'],
            bucket_name=os.environ['S3_RESULTS_BUCKET'],
            key_prefix=f'{os.environ["S3_RESULTS_PREFIX"]}/batch-extract/{dataset_name}',
            role_arn=os.environ['BATCH_INFERENCE_ROLE'],
            max_batch_size=40000,
            max_num_concurrent_batches=1
        )
        indexing_config = IndexingConfig(batch_config=batch_config)

    if doc_store == 's3':
        extracted_docs = S3BasedDocs(
            region=os.environ['AWS_REGION_NAME'],
            bucket_name=os.environ['S3_RESULTS_BUCKET'],
            key_prefix=f'{os.environ["S3_RESULTS_PREFIX"]}/doc-store/{dataset_name}',
            collection_id=None,
            for_jsonl=os.environ.get('BENCHMARK_S3_JSONL', 'false').lower() == 'true'
        )
    else:
        extracted_docs = FileBasedDocs(
            docs_directory=os.path.join(data_dir, dataset_name, 'extracted'),
            collection_id=dataset_name
        )

    logger.info(
        f'Doc store: {type(extracted_docs).__name__} '
        f'(for_jsonl={getattr(extracted_docs, "for_jsonl", "n/a")}) '
        f'collection_id={extracted_docs.collection_id}'
    )

    with (
        GraphStoreFactory.for_graph_store(
            os.environ['GRAPH_STORE'],
            log_formatting=NonRedactedGraphQueryLogFormatting()
        ) as graph_store,
        VectorStoreFactory.for_vector_store(os.environ['VECTOR_STORE']) as vector_store
    ):
        if indexing_config:
            graph_index = LexicalGraphIndex(graph_store, vector_store, indexing_config=indexing_config)
        else:
            graph_index = LexicalGraphIndex(graph_store, vector_store)

        with run_metrics.phase('read_source_documents'):
            docs = SimpleDirectoryReader(input_dir=input_path).load_data()
        logger.info(f'Starting extraction for {len(docs)} documents')

        handler.add_output('num_source_docs', len(docs))

        with run_metrics.phase('extract'):
            graph_index.extract(docs, handler=extracted_docs, show_progress=True)

    with run_metrics.phase('count_extracted_docs'):
        num_extracted = _count_source_docs(extracted_docs)

    handler.add_output('num_extracted_docs', num_extracted)
    handler.add_output('collection_id', extracted_docs.collection_id)
    handler.add_output('extraction_num_workers', GraphRAGConfig.extraction_num_workers)
    handler.add_output(
        'extraction_num_threads_per_worker', GraphRAGConfig.extraction_num_threads_per_worker
    )
    handler.add_output('use_batch', use_batch)
    handler.add_output('doc_store', doc_store)

    # Documents that failed extraction are set to None and logged at debug level
    # rather than raised (batch_extractor_base.py), so a drop is only visible as
    # a shortfall against the input count. Recorded, not asserted on - the
    # expected-count assertion below is what fails the run.
    handler.add_output('num_dropped_docs', len(docs) - num_extracted)

    metrics_dir = os.environ.get(run_metrics.METRICS_DIR_VAR)
    if metrics_dir:
        # Flush this process before summarizing; workers flush at their own exit.
        process_metrics = run_metrics.get_metrics()
        if process_metrics:
            process_metrics.flush()
        handler.add_output('run_metrics', run_metrics.summarize(metrics_dir))

    # Extraction runs in spawn-started children, which carry none of the
    # parent's logging config or handlers, so their retries are only ever
    # visible in the logs. This is the count that covers extraction.
    retry_log_paths = run_metrics.default_retry_log_paths()
    handler.add_output('retry_log_paths', retry_log_paths)
    handler.add_output('retry_counts', run_metrics.count_retries_in_logs(retry_log_paths))

    class BenchmarkExtractAssertions(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls._num_extracted = num_extracted
            cls._expected_num_docs = expected_docs

        def test_extracted_docs_exist(self):
            """At least one document was extracted"""
            self.assertGreater(self._num_extracted, 0)

        def test_expected_doc_count(self):
            """Extracted the expected number of documents"""
            if self._expected_num_docs is not None:
                self.assertEqual(self._num_extracted, self._expected_num_docs)

    handler.run_assertions(BenchmarkExtractAssertions)


class ConcurrentQaBenchmarkExtract(IntegrationTestBase):

    @property
    def description(self):
        return 'Extract propositions and topics from ConcurrentQA documents using batch inference'

    def _run_test(self, handler: IntegrationTestHandler, params: Dict[str, Any]):
        is_prototype = os.environ.get('BENCHMARK_IS_PROTOTYPE')
        dataset_name = 'concurrentqa-prototype' if is_prototype == 'true' else 'concurrentqa'
        expected_docs = 2 if is_prototype == 'true' else 13501
        use_batch = is_prototype != 'true'

        run_benchmark_extract(handler, dataset_name, BENCHMARK_DATA_DIR, expected_docs, use_batch)


class WikihowBenchmarkExtract(IntegrationTestBase):

    @property
    def description(self):
        return 'Extract propositions and topics from WikiHow documents using batch inference'

    def _run_test(self, handler: IntegrationTestHandler, params: Dict[str, Any]):
        run_benchmark_extract(handler, 'wikihow', BENCHMARK_DATA_DIR, expected_docs=5000)


class PgaBenchmarkExtract(IntegrationTestBase):

    @property
    def description(self):
        return 'Extract propositions and topics from PGA documents using batch inference'

    def _run_test(self, handler: IntegrationTestHandler, params: Dict[str, Any]):
        run_benchmark_extract(handler, 'pga', BENCHMARK_DATA_DIR, expected_docs=507)

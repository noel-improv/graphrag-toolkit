# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pipe import Pipe
import multiprocessing
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Optional, Sequence, Any, cast, Callable, Generator, Union


from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.ingestion.pipeline import run_transformations
from llama_index.core.schema import BaseNode, Document

from graphrag_toolkit.lexical_graph.config import GraphRAGConfig


def _init_worker(config_snapshot):
    """Re-apply the parent's GraphRAGConfig scalars in a spawn-started worker.

    spawn re-imports config.py in a clean interpreter, so the GraphRAGConfig
    singleton loses any programmatically-set value and would silently fall back
    to env/None - widening effective permissions (a scoped aws_profile becomes
    the ambient role) and mis-placing data (s3_chunk_store -> None falls back to
    the in-graph chunk store, dropping the intended KMS CMK). Re-applying the
    snapshot keeps workers consistent with the parent.
    """
    GraphRAGConfig.apply_config_snapshot(config_snapshot)


def _sink():
    def _sink_from(generator):
        for item in generator:
            continue
    return Pipe(_sink_from)

sink = _sink()


@contextmanager
def pipeline_executor(num_workers:int):
    """
    The worker pool for one pipeline run, shared across its batches: None when
    one worker runs in the calling process, otherwise a spawn-context
    ProcessPoolExecutor that is shut down when the run ends.

    Starting a worker costs an interpreter start and a package import, so a run
    pays that once per worker rather than once per batch.

    Use "spawn": a forked worker can inherit a held lock (e.g. a logging
    thread's) and deadlock. Spawn starts workers from a clean interpreter,
    which also drops the GraphRAGConfig singleton's programmatically-set
    values - so propagate a picklable snapshot via the worker initializer.
    """
    if num_workers == 1:
        yield None
        return

    config_snapshot = GraphRAGConfig.get_config_snapshot()
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=multiprocessing.get_context('spawn'),
        initializer=_init_worker,
        initargs=(config_snapshot,),
    ) as executor:
        yield executor


def run_pipeline(
    pipeline:IngestionPipeline,
    node_batches:List[List[BaseNode]],
    cache_collection: Optional[str] = None,
    in_place: bool = True,
    num_workers: int = 1,
    executor: Optional[ProcessPoolExecutor] = None,
    **kwargs: Any,
) -> Sequence[BaseNode]:
    """
    Runs the pipeline's transformations over each node batch and yields the
    nodes. One worker runs in the calling process. More than one run on
    ``executor`` when given, so a caller with many batches starts its pool once,
    and on a pool of their own otherwise.
    """
    transform: Callable[[List[BaseNode]], List[BaseNode]] = partial(
        run_transformations,
        transformations=pipeline.transformations,
        in_place=in_place,
        cache=pipeline.cache if not pipeline.disable_cache else None,
        cache_collection=cache_collection,
        **kwargs
    )

    if num_workers == 1:
        for node_batch in node_batches:
            for processed_node in transform(node_batch):
                yield processed_node
        return

    if executor is not None:
        for processed_node_batch in executor.map(transform, node_batches):
            for processed_node in processed_node_batch:
                yield processed_node
        return

    with pipeline_executor(num_workers) as own_executor:
        for processed_node_batch in own_executor.map(transform, node_batches):
            for processed_node in processed_node_batch:
                yield processed_node

def node_batcher(
        num_batches: int, nodes: Union[Sequence[BaseNode], List[Document]]
    ) -> Generator[Union[Sequence[BaseNode], List[Document]], Any, Any]:
        num_nodes = len(nodes)
        batch_size = max(1, int(num_nodes / num_batches))
        if batch_size * num_batches < num_nodes:
             batch_size += 1
        for i in range(0, num_nodes, batch_size):
            yield nodes[i : i + batch_size]

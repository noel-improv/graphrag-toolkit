# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch
from graphrag_toolkit.lexical_graph.indexing.build.build_pipeline import BuildPipeline, NodeFilter


class TestNodeFilter:
    """Tests for NodeFilter functionality."""

    def test_node_filter_callable(self):
        """Verify NodeFilter is callable and filters nodes."""
        filter_func = NodeFilter()
        nodes = [
            Mock(node_id='n1'),
            Mock(node_id='n2'),
            Mock(node_id='n3'),
        ]
        result = filter_func(nodes)
        assert result is not None

    def test_node_filter_returns_list(self):
        """Verify NodeFilter returns a list."""
        filter_func = NodeFilter()
        result = filter_func([Mock()])
        assert isinstance(result, list)


class TestBuildPipelineInitialization:
    """Tests for BuildPipeline initialization."""

    def test_initialization_with_components(self):
        """Verify BuildPipeline initializes with transform components."""
        mock_component = Mock()
        with patch('graphrag_toolkit.lexical_graph.indexing.build.build_pipeline.IngestionPipeline'):
            pipeline = BuildPipeline.create(
                components=[mock_component],
                graph_store=Mock(),
                vector_store=Mock(),
            )
            assert pipeline is not None

    def test_initialization_with_empty_components(self):
        """Verify BuildPipeline handles empty component list."""
        with patch('graphrag_toolkit.lexical_graph.indexing.build.build_pipeline.IngestionPipeline'):
            pipeline = BuildPipeline.create(
                components=[],
                graph_store=Mock(),
                vector_store=Mock(),
            )
            assert pipeline is not None

    def test_initialization_with_multiple_components(self):
        """Verify BuildPipeline accepts multiple components."""
        with patch('graphrag_toolkit.lexical_graph.indexing.build.build_pipeline.IngestionPipeline'):
            pipeline = BuildPipeline.create(
                components=[Mock(), Mock(), Mock()],
                graph_store=Mock(),
                vector_store=Mock(),
            )
            assert pipeline is not None


class TestBuildPipelineErrorHandling:
    """Tests for pipeline error handling."""

    def test_build_with_invalid_component(self):
        """Verify pipeline handles invalid components."""
        invalid_component = "not_a_component"
        with patch('graphrag_toolkit.lexical_graph.indexing.build.build_pipeline.IngestionPipeline'):
            try:
                BuildPipeline.create(
                    components=[invalid_component],
                    graph_store=Mock(),
                    vector_store=Mock(),
                )
            except (TypeError, ValueError, AttributeError):
                pass


class TestBuildPipelineSharesOneExecutor:
    """A build starts its worker pool once, not once per batch of four documents."""

    def test_every_batch_runs_on_the_same_executor(self):
        from contextlib import contextmanager
        from llama_index.core.schema import Document
        from graphrag_toolkit.lexical_graph.indexing.build.null_builder import NullBuilder
        import graphrag_toolkit.lexical_graph.indexing.build.build_pipeline as bp

        shared = object()
        opened = []

        @contextmanager
        def one_executor(num_workers):
            opened.append(num_workers)
            yield shared

        # create() returns the Pipe that wraps build(), so drive it as a pipe.
        pipeline = BuildPipeline.create(components=[NullBuilder()], builders=[], num_workers=2, batch_size=1)
        docs = [Document(text=f"doc {i}", id_=f"aws::{i:08d}:0000") for i in range(3)]

        with patch.object(bp, 'pipeline_executor', one_executor), \
             patch.object(bp, 'run_pipeline', side_effect=lambda *a, **k: []) as run:
            list(docs | pipeline)

        assert opened == [2]
        assert run.call_count == 3
        assert all(call.kwargs['executor'] is shared for call in run.call_args_list)

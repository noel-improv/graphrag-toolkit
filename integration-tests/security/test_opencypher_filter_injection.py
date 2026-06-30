# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security integration test: OpenCypher injection in the lexical-graph filter builder.

Runs the clause emitted by the real filter builder against a live openCypher
engine and shows the fix doing its job: legitimate filters resolve, injection
payloads are inert, and the pre-fix interpolation pattern still breaks out.

Backend: NEO4J_TEST_URI (see conftest.py). Skips when unset.
"""

import pytest
from llama_index.core.vector_stores.types import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters,
)
from graphrag_toolkit.lexical_graph.storage.graph.graph_utils import (
    parse_metadata_filters_recursive,
)

VALUE_PAYLOAD = "' OR '1'='1"


@pytest.fixture(autouse=True)
def _seed(neo4j_driver):
    with neo4j_driver.session() as s:
        s.run('MATCH (n) DETACH DELETE n')
        s.run("CREATE (:Source {category:'tech'}), (:Source {category:'science'})")


def _clause(key, value, operator=FilterOperator.EQ):
    return parse_metadata_filters_recursive(MetadataFilters(
        filters=[MetadataFilter(key=key, value=value, operator=operator)],
        condition=FilterCondition.AND))


def _run(neo4j_driver, clause):
    query = f'MATCH (source:Source) WHERE {clause} RETURN source.category AS c ORDER BY c'
    with neo4j_driver.session() as s:
        return [r['c'] for r in s.run(query)]


def test_legit_filter_resolves(neo4j_driver):
    assert _run(neo4j_driver, _clause('category', 'tech')) == ['tech']


def test_injected_value_matches_nothing(neo4j_driver):
    assert _run(neo4j_driver, _clause('category', VALUE_PAYLOAD)) == []


def test_injected_key_matches_nothing(neo4j_driver):
    assert _run(neo4j_driver, _clause("x' OR '1'='1", 'tech')) == []


def test_pre_fix_pattern_would_break_out(neo4j_driver):
    """Red-state: the old interpolated filter bypasses on the same engine."""
    old_clause = "(source.category = '' OR '1'='1')"
    assert _run(neo4j_driver, old_clause) == ['science', 'tech']

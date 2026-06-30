# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security integration test: property-name injection in the byokg-rag Neptune store.

Builds the property fragment with the real _escape_cypher_label helper and runs
it against a live openCypher engine: a backtick-quoted property is treated as a
literal key, an injection-laden name is inert, and the pre-fix raw interpolation
raises a syntax error on the engine.

Backend: NEO4J_TEST_URI (see conftest.py). Skips when unset.
"""

import pytest
from neo4j.exceptions import CypherSyntaxError
from graphrag_toolkit.byokg_rag.graphstore.neptune import _escape_cypher_label


@pytest.fixture(autouse=True)
def _seed(neo4j_driver):
    with neo4j_driver.session() as s:
        s.run('MATCH (n) DETACH DELETE n')
        s.run("CREATE (:N {name:'keep'})")


def _run_fixed(neo4j_driver, prop):
    fragment = f"n.`{_escape_cypher_label(prop)}`"
    query = f'MATCH (n:N) WHERE {fragment} IN $ids RETURN n.name AS name'
    with neo4j_driver.session() as s:
        return [r['name'] for r in s.run(query, ids=['keep'])]


def test_legit_property_resolves(neo4j_driver):
    assert _run_fixed(neo4j_driver, 'name') == ['keep']


def test_injected_property_name_is_inert(neo4j_driver):
    assert _run_fixed(neo4j_driver, 'a`b') == []
    assert _run_fixed(neo4j_driver, "x` OR n.secret=1 OR `") == []


def test_pre_fix_pattern_would_break_out(neo4j_driver):
    """Red-state: the old raw property interpolation breaks the query on the engine."""
    query = 'MATCH (n:N) WHERE n.a`b IN $ids RETURN n.name AS name'
    with neo4j_driver.session() as s:
        with pytest.raises(CypherSyntaxError):
            list(s.run(query, ids=['keep']))

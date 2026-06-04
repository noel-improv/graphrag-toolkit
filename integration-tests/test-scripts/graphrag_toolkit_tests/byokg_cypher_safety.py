# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from typing import Dict, Any

from graphrag_toolkit_tests.integration_test_base import IntegrationTestBase
from graphrag_toolkit_tests.integration_test_handler import IntegrationTestHandler

from graphrag_toolkit.byokg_rag.graphstore import (
    NeptuneAnalyticsGraphStore,
    NeptuneDBGraphStore,
)
from graphrag_toolkit.byokg_rag.graphstore.neptune import _escape_cypher_label


# A node label and relationship type carrying a Cypher breakout payload. If a
# sink interpolates one into a backtick-quoted identifier without escaping, the
# appended clause deletes the canary; with escaping the canary survives.
CANARY_LABEL = '__CypherSafetyCanary__'
CANARY_ID = 'cypher-safety-canary'
MALICIOUS_NODE_LABEL = (
    'EvilNode`) MATCH (c:`__CypherSafetyCanary__`) DETACH DELETE c //'
)
MALICIOUS_EDGE_TYPE = (
    'EvilEdge`]->() MATCH (c:`__CypherSafetyCanary__`) DETACH DELETE c //'
)
_TMP_IDS = [CANARY_ID, 'evil-node', 'evil-edge-a', 'evil-edge-b']


class BYOKGCypherInjectionSafety(IntegrationTestBase):
    """Insert a backtick-bearing label into a live Neptune graph, run the
    schema-discovery path, and confirm no injected clause executes."""

    @property
    def description(self):
        return 'Schema discovery escapes backticks in dynamic Cypher labels'

    def _make_graph_store(self):
        region = os.environ['AWS_REGION_NAME']
        graph_store_id = os.environ['GRAPH_STORE']

        if graph_store_id.startswith('neptune-graph://'):
            graph_identifier = graph_store_id[len('neptune-graph://'):]
            store = NeptuneAnalyticsGraphStore(
                graph_identifier=graph_identifier, region=region
            )
            return 'analytics', store

        if graph_store_id.startswith('neptune-db://'):
            endpoint = graph_store_id[len('neptune-db://'):]
            if not endpoint.startswith('https://'):
                endpoint = f'https://{endpoint}'
            return 'db', NeptuneDBGraphStore(endpoint_url=endpoint, region=region)

        raise ValueError(
            "Invalid graph store id. Expected 'neptune-graph://' or "
            f"'neptune-db://', but received {graph_store_id}."
        )

    def _canary_count(self, graph_store):
        rows = graph_store.execute_query(
            f'MATCH (c:`{CANARY_LABEL}`) RETURN count(c) AS n'
        )
        return rows[0]['n'] if rows else 0

    def _seed_payload(self, graph_store):
        # The label/type are escaped here on the write path so the payload is
        # stored as data; the read path (under test) is what must re-escape it.
        graph_store.execute_query(
            f'CREATE (c:`{CANARY_LABEL}` {{id: $id}})',
            parameters={'id': CANARY_ID},
        )
        graph_store.execute_query(
            f'CREATE (n:`{_escape_cypher_label(MALICIOUS_NODE_LABEL)}` {{id: $id}})',
            parameters={'id': 'evil-node'},
        )
        graph_store.execute_query(
            f'CREATE (a:`__CypherSafetyTmp__` {{id: $aid}})'
            f'-[:`{_escape_cypher_label(MALICIOUS_EDGE_TYPE)}`]->'
            f'(b:`__CypherSafetyTmp__` {{id: $bid}})',
            parameters={'aid': 'evil-edge-a', 'bid': 'evil-edge-b'},
        )

    def _exercise_sinks(self, engine, graph_store):
        """Drive the escaped schema-discovery sinks for this engine and return
        the labels the path observed."""
        if engine == 'db':
            # get_schema() -> _refresh_schema() -> _get_node_properties /
            # _get_edge_properties / _get_triples, each interpolating a label
            # read back from the graph (including the payload labels seeded above).
            schema = graph_store.get_schema()
            return list(schema.get('nodeLabels', []))

        # Analytics: pg_schema() does not interpolate labels, so the payload
        # flows in as the node_type argument to the shared sinks instead.
        graph_store.nodes(node_type=MALICIOUS_NODE_LABEL)
        graph_store.get_node_text_for_embedding_input(
            node_embedding_text_props={MALICIOUS_NODE_LABEL: ['name']},
            group_by_node_label=True,
        )
        return [MALICIOUS_NODE_LABEL]

    def _run_test(self, handler: IntegrationTestHandler, params: Dict[str, Any]):

        engine, graph_store = self._make_graph_store()

        self._seed_payload(graph_store)
        canary_before = self._canary_count(graph_store)

        schema_error = None
        discovered_labels = []
        try:
            discovered_labels = self._exercise_sinks(engine, graph_store)
        except Exception as e:
            schema_error = e

        canary_after = self._canary_count(graph_store)

        # Best-effort cleanup; matching by id is label-agnostic.
        try:
            graph_store.execute_query(
                'MATCH (n) WHERE n.id IN $ids DETACH DELETE n',
                parameters={'ids': _TMP_IDS},
            )
        except Exception as e:
            handler.add_exception(e)

        handler.add_output('engine', engine)
        handler.add_output('canary_before', canary_before)
        handler.add_output('canary_after', canary_after)
        handler.add_output(
            'schema_error', str(schema_error) if schema_error else None
        )
        handler.add_output('discovered_labels', discovered_labels)

        class CypherInjectionSafetyAssertions(unittest.TestCase):

            @classmethod
            def setUpClass(cls):
                cls._engine = engine
                cls._canary_before = canary_before
                cls._canary_after = canary_after
                cls._schema_error = schema_error
                cls._discovered_labels = discovered_labels

            def test_setup_created_canary(self):
                """Canary node exists before schema discovery runs"""
                self.assertEqual(self._canary_before, 1)

            def test_schema_discovery_did_not_error(self):
                """Schema discovery completes without raising"""
                self.assertIsNone(self._schema_error)

            def test_canary_survives_injection_payload(self):
                """Canary still present: no injected DETACH DELETE executed"""
                self.assertEqual(self._canary_after, 1)

            def test_payload_label_flowed_through_discovery(self):
                """The backtick-bearing label was processed as data, not code"""
                self.assertIn(MALICIOUS_NODE_LABEL, self._discovered_labels)

        handler.run_assertions(CypherInjectionSafetyAssertions)

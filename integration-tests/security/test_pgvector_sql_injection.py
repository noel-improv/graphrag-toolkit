# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security integration test: SQL injection in the PGVector store.

Drives the real PGIndex sinks against a live Postgres+pgvector engine and shows
the fix doing its job: legitimate filters resolve, injection payloads are inert,
and the pre-fix interpolation pattern still breaks out on the same engine.

Backend: PGVECTOR_TEST_DSN (see conftest.py). Skips when unset.
"""

import json
import numpy as np
import pytest

import graphrag_toolkit.lexical_graph.storage.vector.pg_vector_indexes as pvi
from graphrag_toolkit.lexical_graph.tenant_id import TenantId
from graphrag_toolkit.lexical_graph.metadata import FilterConfig
from llama_index.core.schema import QueryBundle
from llama_index.core.vector_stores.types import (
    FilterCondition, FilterOperator, MetadataFilter, MetadataFilters,
)

VALUE_PAYLOAD = "' OR '1'='1"
ID_PAYLOAD = "x') OR '1'='1"


@pytest.fixture
def index(pg_kwargs):
    from pgvector.psycopg2 import register_vector

    idx = pvi.PGIndex.model_construct(
        index_name='chunk', database=pg_kwargs['dbname'], schema_name='graphrag',
        host=pg_kwargs['host'], port=pg_kwargs['port'], username=pg_kwargs['user'],
        password=pg_kwargs['password'], dimensions=3, embed_model=None,
        enable_iam_db_auth=False, writeable=True, tenant_id=TenantId(), initialized=False,
    )
    table = idx.underlying_index_name()
    conn = idx._get_connection()
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f'TRUNCATE graphrag.{table};')  # nosec B608 - test-local identifier
    rows = [
        ('keep1', {'source': {'metadata': {'category': 'tech', 'count': 5}, 'versioning': {}}}, [1.0, 0.0, 0.0]),
        ('keep2', {'source': {'metadata': {'category': 'science', 'count': 1}, 'versioning': {}}}, [0.0, 1.0, 0.0]),
    ]
    for cid, meta, emb in rows:
        cur.execute(
            f'INSERT INTO graphrag.{table} (chunkId, value, metadata, embedding) '  # nosec B608 - test-local identifier
            'VALUES (%s, %s, %s, %s);',
            (cid, cid, json.dumps(meta), np.array(emb)),
        )
    conn.close()
    idx._table = table
    return idx


def _top_k(index, filter_config):
    bundle = QueryBundle(query_str='q', embedding=[1.0, 0.0, 0.0])
    orig = pvi.to_embedded_query
    pvi.to_embedded_query = lambda qb, m: bundle
    try:
        return index.top_k(bundle, top_k=10, filter_config=filter_config)
    finally:
        pvi.to_embedded_query = orig


def _eq(key, value):
    return FilterConfig(source_filters=MetadataFilters(
        filters=[MetadataFilter(key=key, value=value, operator=FilterOperator.EQ)],
        condition=FilterCondition.AND))


def _row_count(pg_kwargs, table, where=''):
    import psycopg2
    conn = psycopg2.connect(**pg_kwargs)
    cur = conn.cursor()
    cur.execute(f'SELECT count(*) FROM graphrag.{table} {where};')  # nosec B608 - test-local
    n = cur.fetchone()[0]
    conn.close()
    return n


def test_legit_text_filter_resolves(index):
    results = _top_k(index, _eq('category', 'tech'))
    assert [r['source']['metadata']['category'] for r in results] == ['tech']


def test_injected_filter_value_matches_nothing(index):
    assert _top_k(index, _eq('category', VALUE_PAYLOAD)) == []


def test_injected_id_does_not_delete(index, pg_kwargs):
    index.delete_embeddings(ids=[ID_PAYLOAD])
    assert _row_count(pg_kwargs, index._table) == 2


def test_injected_id_does_not_update_versioning(index, pg_kwargs):
    index.update_versioning(versioning_timestamp=999, ids=[ID_PAYLOAD])
    assert _row_count(pg_kwargs, index._table, 'WHERE valid_to = 999') == 0


def test_pre_fix_pattern_would_break_out(index, pg_kwargs):
    """Red-state: the old interpolated filter bypasses on the same engine, so the
    inert result above is the fix working, not the engine being lenient."""
    import psycopg2
    conn = psycopg2.connect(**pg_kwargs)
    cur = conn.cursor()
    old_where = "(((metadata->'source'->'metadata'->>'category')::text = '%s'))" % VALUE_PAYLOAD
    cur.execute(f"SELECT chunkId FROM graphrag.{index._table} WHERE {old_where};")  # nosec B608 - red-state demo
    bypassed = [r[0] for r in cur.fetchall()]
    conn.close()
    assert len(bypassed) == 2, 'expected the pre-fix interpolation to bypass the filter'

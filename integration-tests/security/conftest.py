# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the security integration suite.

Each backend is gated on an environment variable so the suite skips cleanly
when no live engine is available (local dev without containers) and runs in CI
against service containers. See README.md.
"""

import os
import pytest


def _pg_kwargs():
    dsn = os.environ.get('PGVECTOR_TEST_DSN')
    if not dsn:
        return None
    out = {}
    for part in dsn.split():
        key, value = part.split('=', 1)
        out[key] = int(value) if key == 'port' else value
    return out


@pytest.fixture(scope='session')
def pg_kwargs():
    """psycopg2 connection kwargs for the live Postgres+pgvector engine."""
    kw = _pg_kwargs()
    if kw is None:
        pytest.skip('PGVECTOR_TEST_DSN not set')
    return kw


@pytest.fixture(scope='session', autouse=True)
def _pg_extension_and_schema():
    """Ensure the pgvector extension and graphrag schema exist before any test
    that uses the Postgres backend. No-op when the backend is not configured."""
    kw = _pg_kwargs()
    if kw is None:
        return
    import psycopg2
    conn = psycopg2.connect(**kw)
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    cur.execute('CREATE SCHEMA IF NOT EXISTS graphrag;')
    cur.close()
    conn.close()


@pytest.fixture
def neo4j_driver():
    """A connected neo4j driver for the live openCypher engine."""
    uri = os.environ.get('NEO4J_TEST_URI')
    if not uri:
        pytest.skip('NEO4J_TEST_URI not set')
    from neo4j import GraphDatabase
    user = os.environ.get('NEO4J_TEST_USER', 'neo4j')
    password = os.environ.get('NEO4J_TEST_PASSWORD', 'testpassword123')
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    yield driver
    driver.close()

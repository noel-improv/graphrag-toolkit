# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the security integration suite.

Each backend is gated on an environment variable so the suite skips when no live
engine is available and runs in CI against service containers. See README.md.
"""

import os
import time
import pytest


def _pg_kwargs():
    """Parse PGVECTOR_TEST_DSN into psycopg2 connect kwargs, or None when unset."""
    dsn = os.environ.get('PGVECTOR_TEST_DSN')
    if not dsn:
        return None
    import psycopg2.extensions
    kwargs = psycopg2.extensions.parse_dsn(dsn)
    if 'port' in kwargs:
        kwargs['port'] = int(kwargs['port'])
    return kwargs


@pytest.fixture(scope='session')
def pg_kwargs():
    """psycopg2 connect kwargs for the live Postgres+pgvector engine."""
    kwargs = _pg_kwargs()
    if kwargs is None:
        pytest.skip('PGVECTOR_TEST_DSN not set')
    return kwargs


@pytest.fixture(scope='session', autouse=True)
def _pg_extension_and_schema():
    """Ensure the pgvector extension and graphrag schema exist before any test
    that uses the Postgres backend. A no-op when the backend is not configured."""
    kwargs = _pg_kwargs()
    if kwargs is None:
        return
    import psycopg2
    conn = psycopg2.connect(**kwargs)
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    cur.execute('CREATE SCHEMA IF NOT EXISTS graphrag;')
    cur.close()
    conn.close()


@pytest.fixture(scope='session')
def neo4j_driver():
    """A connected neo4j driver for the live openCypher engine.

    Retries verify_connectivity because a service container can report healthy
    over HTTP before Bolt accepts authenticated connections.
    """
    uri = os.environ.get('NEO4J_TEST_URI')
    if not uri:
        pytest.skip('NEO4J_TEST_URI not set')
    from neo4j import GraphDatabase
    user = os.environ.get('NEO4J_TEST_USER', 'neo4j')
    password = os.environ.get('NEO4J_TEST_PASSWORD', 'testpassword123')
    driver = GraphDatabase.driver(uri, auth=(user, password))
    last_error = None
    for _ in range(30):
        try:
            driver.verify_connectivity()
            break
        except Exception as error:
            last_error = error
            time.sleep(2)
    else:
        driver.close()
        raise last_error
    yield driver
    driver.close()

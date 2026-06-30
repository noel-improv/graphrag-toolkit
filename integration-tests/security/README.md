# Security integration tests

Grouped integration suite that runs each query-injection fix against a live
engine and shows the fix working: a legitimate query resolves, the injection
payload is inert, and the pre-fix interpolation pattern still breaks out on the
same engine. AppSec runs this on each release to confirm the fixes hold.

One test module per fix; add a module per future security fix rather than
scattering canaries across the package test suites.

| Module | Fix | Backend |
|--------|-----|---------|
| `test_pgvector_sql_injection.py` | SQL injection in the PGVector store filters/id lists | Postgres + pgvector |
| `test_opencypher_filter_injection.py` | OpenCypher injection in the lexical-graph filter builder | openCypher (Neo4j) |
| `test_byokg_property_injection.py` | Property-name injection in the byokg-rag Neptune store | openCypher (Neo4j) |

## Backends

Each backend is gated on an environment variable; the suite skips cleanly when a
backend is not configured.

- `PGVECTOR_TEST_DSN` — e.g. `host=localhost port=5432 dbname=testdb user=test password=test`
- `NEO4J_TEST_URI` — e.g. `bolt://localhost:7687` (plus `NEO4J_TEST_USER` / `NEO4J_TEST_PASSWORD`)

CI provides these via service containers (see
`.github/workflows/security-integration-tests.yml`). Neo4j is used as an
openCypher-compatible stand-in for Neptune; the escaping under test is standard
Cypher honored by both. Neptune-specific validation belongs in the CloudFormation
suite, not in this per-release gate.

## Run locally

```bash
# Postgres + pgvector
docker run -d --name pgv -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=testdb -p 5432:5432 pgvector/pgvector:pg16
# Neo4j
docker run -d --name neo4j -e NEO4J_AUTH=neo4j/testpassword123 \
  -p 7687:7687 -p 7474:7474 neo4j:5.25-community

pip install ./lexical-graph ./byokg-rag
pip install -r integration-tests/security/requirements.txt

export PGVECTOR_TEST_DSN="host=localhost port=5432 dbname=testdb user=test password=test"
export NEO4J_TEST_URI="bolt://localhost:7687"
pytest integration-tests/security/ -v
```

# Benchmark Reproducibility

This pins the exact configuration behind the published benchmark baseline so the
results can be reproduced. For the full run procedure (stack deploy, S3 data
layout, the per-retriever loop on the notebook), see [BENCHMARKING.md](./BENCHMARKING.md).
This document records the specific knobs that fix a run.

## Baseline configuration

| Setting | Value |
|---|---|
| Response LLM | Claude Haiku 4.5 (`us.anthropic.claude-haiku-4-5-20251001-v1:0`), set via `TEST_RESPONSE_LLM` |
| Reranker | TF-IDF |
| Graph store | Amazon Neptune `db.r8g.2xlarge` |
| Notebook | `ml.m5.xlarge` (CUAD), `ml.m5.4xlarge` (ConcurrentQA, WikiHow) |

The query harness defaults `TEST_RESPONSE_LLM` to Sonnet 4.6, so the baseline must
set it to the Haiku model id above to reproduce these numbers.

### Datasets

| Dataset | Documents | QA pairs |
|---|---|---|
| CUAD | 510 | 500 |
| ConcurrentQA | 13,501 | 400 |
| WikiHow | 5,000 | 300 |
| PGA | 507 | 240 |

## Extraction config per run

The graph each run queries is built in the extract/build phase, not the query
harness, so the extraction config is fixed there. The knobs the benchmark sets:

| Setting | Value | Source |
|---|---|---|
| Extraction LLM | `us.anthropic.claude-sonnet-4-6` (set via `TEST_EXTRACTION_LLM`) | `benchmark_extract.py` |
| Extraction batch size | `15000` | `benchmark_extract.py` |
| Extraction workers | `2` | `benchmark_extract.py` |
| Batch inference max | `40000` (when `use_batch`) | `benchmark_extract.py` |

Chunk size and embedding model are not overridden by the benchmark, so they use
the toolkit defaults at the pinned package version. The resulting per-dataset graph
stats (chunks, topics, statements, facts, entities) live in a table in the Benchmark
Results Quip, computed from a separate run rather than captured live in the harness.

## Retriever hyperparameters

Each run records a `retriever_config` block into
`benchmark-results/<dataset>/<retriever_id>/metrics_summary.json`, so a run
self-documents its retriever configuration. The hyperparameters per retriever:

| Retriever | Hyperparameters |
|---|---|
| `topic_based`, `entity_based`, `chunk_based`, `entity_network`, `chunk_based_semantic` | `reranker=tfidf`, `vss_top_k=10`, `max_search_results=5`, `max_statements=200`, `derive_subqueries=False` |
| `topic-beam-chunk_only` | `ChunkCosineSimilaritySearch top_k=100`; `SemanticChunkBeamGraphSearch max_depth=3, beam_width=10` |
| `semantic-path_weighted` | `RerankingBeamGraphSearch max_depth=3, beam_width=10` |
| `agentic` | `max_iterations=3` |
| `byokg_agentic` | `max_iterations=2` |
| `traversal`, `semantic_guided` | retrieval-library defaults (no harness overrides) |

Notes:

- The sub-retriever values come from `_SUB_RETRIEVER_PROCESSOR_ARGS` in
  `test-scripts/graphrag_toolkit_tests/benchmark_utils/retriever_factory.py`.
- The beam values for `topic-beam-chunk_only` and `semantic-path_weighted` are the
  search-class defaults at the pinned package version. The harness passes no
  explicit overrides for these, so they reproduce via the pinned dependency.
- `traversal` and `semantic_guided` likewise run on library defaults.

## Reproducing the baseline

Build the graph for a dataset per [BENCHMARKING.md](./BENCHMARKING.md), then run the
retrievers on the notebook with the pinned model. Replace `<Dataset>` with one of
`Cuad`, `ConcurrentQa`, `Wikihow`, `Pga`:

```bash
source .env.testing && source .env
export TEST_RESPONSE_LLM=us.anthropic.claude-haiku-4-5-20251001-v1:0

for RETRIEVER in topic_based entity_based chunk_based entity_network \
                 chunk_based_semantic semantic_guided \
                 topic-beam-chunk_only semantic-path_weighted; do
  export BENCHMARK_RETRIEVER=$RETRIEVER
  export TESTS="benchmark_query.<Dataset>BenchmarkQuery benchmark_evaluate.<Dataset>BenchmarkEvaluate"
  echo "=== $RETRIEVER ==="
  python test_suite.py
done
```

Each run writes `metrics_summary.json` (including the `retriever_config` block) and
`responses.jsonl` under `benchmark-results/<dataset>/<retriever_id>/`. To inspect
the recorded configuration:

```bash
jq '.retriever_config' benchmark-results/cuad/topic_based/metrics_summary.json
```

## Open items

- **Corpus counts live in the Quip.** Per-dataset graph stats (chunks, topics,
  statements, facts, entities) for the benchmarked datasets go in the Benchmark
  Results Quip from a separate run, not in the live benchmark output (avoids the
  per-run latency).
- **Token data incomplete.** ConcurrentQA and PGA token totals were not captured for
  the published run because of a Pydantic v2 issue in the token tracker, since fixed
  for future runs.

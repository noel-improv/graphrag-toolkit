# Extraction benchmark runbook

Measures two claims: that parallel extraction cuts wall time, and that moving chunk text
off the graph reclaims memory. Run from `validation/chunk-store-combined`, which is the
only branch carrying both the connection-pool fix (#420) and the S3 chunk store. Without
the pool fix, S3 read concurrency is capped at botocore's default of 10 and the numbers
are wrong by roughly 8x.

## Phase 0 gates the rest

Nothing else runs until one short run proves the counters fire. A speed-up measured
without retry counts can't be told apart from a run that silently dropped documents.

```bash
export BENCHMARK_METRICS_DIR=run-metrics
export BENCHMARK_IS_PROTOTYPE=true
```

Then run any extract test and check the result JSON has non-zero `num_source_docs`, a
`num_dropped_docs` of 0, and a `retry_counts` block. If `retry_counts` is empty on a run
that logged retry warnings, the log paths are wrong - set `BENCHMARK_RETRY_LOGS`.

## Knobs

| Variable | Default | Effect |
|---|---|---|
| `EXTRACTION_NUM_WORKERS` | 2 | Extraction processes. Capped at CPU count in `extraction_pipeline.py` |
| `EXTRACTION_NUM_THREADS_PER_WORKER` | 4 | Threads per worker (see caveat below) |
| `BENCHMARK_METRICS_DIR` | unset | Enables phase timing and counters |
| `BENCHMARK_RETRY_LOGS` | unset | Comma-separated log paths to scrape for retries |
| `BENCHMARK_DOC_STORE` | `file` | `file` or `s3` |
| `S3_CHUNK_STORE` | unset | `s3://bucket/prefix` moves chunk text off the graph |

## Phases

| Phase | Corpus | Runs | Answers |
|---|---|---|---|
| 0 | prototype | 1 | Do the counters fire |
| 1 | subset | 8 | Threads 1/2/4/8/16/32/64/128, one worker. Where threads stop paying |
| 2 | full 5,000 | 4 | Baseline + best thread count, x2. Speed-up, with doc count unchanged |
| 3 | full 5,000 | 2 | Two workers, one shared S3 store. Does sharing work |
| 4 | full 5,000 | 4 | Storage A/B, arms interleaved: doc store and chunk store, local vs S3 |

Instance size stays fixed across every run so time is the only cost variable. Phase 4
alternates arms rather than batching them, because variance between hours on the same
instance is larger than the effect being measured.

Compute the Bedrock ceiling before reading phase 1: quota TPM divided by measured tokens
per document. If throughput flattens at that ceiling while throttles climb, the run is
quota-bound and the answer is the cheapest thread count that reaches the ceiling, not a
scalability curve. Little's Law is the check - where concurrency stops matching
throughput times latency, the threads are asleep in retry backoff and the curve is an
artifact of the backoff, not of the system.

## Three things this setup cannot see

**Batch inference ignores the thread knob.** `use_batch=True` selects
`BatchLLMPropositionExtractorSync` and `BatchTopicExtractorSync`, which submit Bedrock
batch jobs. `extraction_num_threads_per_worker` never reaches them - it feeds
`TopicExtractor`/`LLMPropositionExtractor` on the on-demand path, plus the S3 read and
write pools. A thread sweep under batch inference measures S3 IO and nothing else, so
phase 1 has to run with `use_batch=False`. The harness also pins
`max_num_concurrent_batches=1`, so the batch path is serial regardless.

**The thread knob is shared.** One number sizes LLM concurrency, the S3 doc-store pools,
and `S3ChunkStore.get_batch`. A sweep moves all three at once, so a knee can't be
attributed to one of them without a second run holding the others fixed. AN-3478 asked
for a listing-specific pool, which does not exist.

**Retry counts are partial.** Three layers retry with different limits - `max_retries=50`
on the LLM object, botocore's `max_attempts=2` in `llm_cache.py`, and tenacity's
monkeypatched decorator in `bedrock_utils.py`. Only the tenacity layer emits a log line,
so botocore's retries are invisible. Report these as observed retries. Throttle count is
the sounder saturation signal.

## Why retries are scraped from logs rather than counted in process

Extraction transformations run in child processes started with spawn
(`pipeline_utils.py`), and that holds even at `num_workers=1`. A spawn child re-imports
from scratch: it inherits no logging config and no handlers, so an in-process counter in
the parent sees nothing. Measured directly - two child retries produced 0 handlers in the
child and 0 counts in the parent, while both warnings still reached the parent's stderr.

So the counts come from the logs. The parent's own warnings go to
`test-logs/<NN>-<Name>.log`; children have no file handler and fall back to stderr, which
the CFN template captures because it starts the suite under `screen -L` into
`screenlog.0`. Both are scraped.

`fm_observability.py` is not used for this. It already counts LLM calls, durations and
tokens, but it installs its queue and callback handlers in the parent only, so it has the
same blind spot - it works for the query phase, which is single-process, and sees nothing
from extraction.

## Reading the output

`run_metrics` sums counters across processes but reports phase timings as total, max and
process count rather than a sum. Workers overlap, so summing their spans would exceed
wall time; the max is the number that compares against it.

p99 query latency needs no code change - `benchmark_query.py` already writes per-query
latencies to `responses.jsonl`, and `metrics_summary.py` simply stops aggregating at p95.
Post-process the JSONL.

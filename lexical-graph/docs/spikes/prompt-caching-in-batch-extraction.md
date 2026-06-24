# Spike: Prompt caching in Bedrock batch extraction

Status: confirmed — documentation finding and an empirical run both agree (see Verification
status). Closed.
Scope: batch extraction API only (non-batch/on-demand extraction is out of scope)
Reference: https://github.com/awslabs/graphrag-toolkit/issues/326 (DRAFT)

## TL;DR

Amazon Bedrock does not support prompt caching with the batch inference API. Prompt
caching is an on-demand-only feature. The toolkit's batch extractors submit work through
`CreateModelInvocationJob` (S3-in, S3-out, asynchronous), so there is no place to attach a
cache checkpoint and no caching to gain. No change to the batch extraction path can deliver
the token-cost or latency improvement this spike set out to find.

The caching opportunity is real, but it lives on the non-batch (on-demand) extraction path,
which this ticket explicitly excludes.

## Verification status

The finding above rests on authoritative AWS documentation (the prompt-caching user guide,
the batch-inference limitations page, and the `CreateModelInvocationJob` API reference),
cross-checked across two independent searches. It is now also empirically confirmed — see
"Empirical confirmation" below.

The empirical result is stronger than the docs imply. Batch inference does not merely ignore
a `cache_control` marker; it hard-rejects every record that carries one with an
`errorCode 400`. The identical request body caches normally on the on-demand path, isolating
the batch API as the sole cause. Earlier draft token estimates have been removed; this
document asserts only measured numbers.

## What was investigated

The question was whether prompt caching could cut token cost or latency for batch
extraction, using the benchmarking suite as the baseline for comparison. Batch extraction
runs the same large extraction prompt against every chunk in a dataset (thousands of
records), so the static instruction block is identical on every request. That is the
classic shape prompt caching is built for: a stable prefix reused across many calls.

Issue #326 (a DRAFT) assumes batch inference supports this — its Proposed Solution point 2
states caching works "when the system prompt is consistent across records," while hedging
"verify batch format compatibility," and its own Current State note says the per-record
JSONL format "doesn't include cache hints." That assumption is what this spike tests.

## How batch extraction works today

The extractors build one JSONL record per node and hand the whole file to a Bedrock batch
job:

- `BatchExtractorBase._process_single_batch` writes each record, uploads the file to S3, and
  calls `create_and_run_batch_job`
  (`lexical-graph/src/graphrag_toolkit/lexical_graph/indexing/extract/batch_extractor_base.py`).
- `create_and_run_batch_job` calls `bedrock_client.create_model_invocation_job(...)` and
  polls `get_model_invocation_job` until the job reaches a terminal state
  (`lexical-graph/src/graphrag_toolkit/lexical_graph/indexing/utils/batch_inference_utils.py:133`).
- Each record's `modelInput` is built by `get_request_body`
  (`batch_inference_utils.py:68`) — for `anthropic.claude` models it produces
  `{anthropic_version, messages, max_tokens, temperature}`.

Every record carries the full extraction prompt. For proposition extraction the static
instruction block precedes the per-node `{source_info}` and `{text}`
(`indexing/prompts.py:4`); for topic extraction the static block precedes `{text}` and the
(usually empty) preferred-value lists (`indexing/prompts.py:57`). In both prompts the
instruction block is byte-identical across every record in a run.

## The blocking finding

Bedrock prompt caching is restricted to on-demand inference. From the AWS prompt-caching
documentation, verbatim:

> Prompt caching is only supported for on-demand inference endpoints. It is not supported
> with the batch inference API.

The supported surfaces are the `Converse`/`ConverseStream` APIs, the
`InvokeModel`/`InvokeModelWithResponseStream` APIs, Bedrock Prompt Management, and the
console playground. The batch inference API (`CreateModelInvocationJob`) is not on that
list, and the note above is explicit that it is excluded.

This is architectural, not an oversight in the request shape. Prompt caching writes a cache
entry on the first request and serves reads from it within an ephemeral TTL (5 minutes for
most models, 1 hour for a few Claude models). Batch inference is asynchronous and
S3-mediated: records are processed by Bedrock's batch fleet with no ordering or temporal
guarantee, and the batch JSONL `modelInput` schema has no field for a cache checkpoint. A
`cache_control` block placed inside a batch record's `modelInput` is not honored — in
practice the batch API rejects the record outright (see Empirical confirmation).

A related note in the same AWS doc says the 1-hour TTL "is useful for ... batch processing
scenarios." That refers to client-side batching of on-demand calls over time (many
`InvokeModel`/`Converse` calls that reuse a warm cache), not the Bedrock batch inference
API. The two are different mechanisms; the explicit exclusion above governs
`CreateModelInvocationJob`.

## What this means for the batch path

No production change. There is nothing to add to `get_request_body` or the batch extractors
that would enable caching, because Bedrock will not cache a batch job's input regardless of
how the record is structured. Adding `cache_control` markers to batch records would be inert
at best and confusing to future maintainers at worst.

Batch inference already carries its own cost reduction (roughly 50% off on-demand token
rates), which is the lever the batch path actually has. Prompt caching is a separate,
mutually exclusive lever that only the on-demand path can pull.

## Where caching does apply: the non-batch path (out of scope here)

Prompt caching is supported on the on-demand extraction path (the `LLMPropositionExtractor`
and `TopicExtractor` invoked via `Converse`/`InvokeModel`). Evaluating it there is the
natural follow-up, and the ticket already names it as a backlog item. The sizing evidence
below is provided so that work can be scoped; this spike does not implement it.

Two factors decide whether caching pays off on the on-demand path:

1. **The static prefix must clear the per-model minimum token count for a cache checkpoint.**
   Per the AWS table, Claude Sonnet 4.6 and Claude 3.7 Sonnet require 1,024 tokens per
   checkpoint; Claude Opus 4.5/4.6, Sonnet 4.5, and Haiku 4.5 require 4,096. Below the
   minimum, inference still succeeds but nothing is cached. (Measured caveat: on
   `us.anthropic.claude-sonnet-4-6` the effective minimum behaved like ~4,096, not the listed
   1,024 — a ~1,400-token prefix did not cache. Size against the measured floor, not the
   table.)

2. **The static prefix must be reused often enough to stay within the TTL** (5 minutes
   default; 1 hour on Opus 4.5 / Sonnet 4.5 / Haiku 4.5). Extraction over a large dataset
   reuses the same prefix continuously, which fits.

Two structural facts hold regardless of measurement:

- The instruction block leads both prompts, so a single cache checkpoint at the end of the
  instructions (before the per-node `{source_info}`/`{text}`) would capture the reusable
  span. Bedrock's simplified cache management for Claude (one breakpoint at the end of static
  content, ~20-block lookback) fits this layout.
- Neither `EXTRACT_PROPOSITIONS_PROMPT` nor `EXTRACT_TOPICS_PROMPT` contains few-shot
  examples — the static content is instructions only. Sizing the cacheable prefix means
  measuring the instruction block, not a system-prompt-plus-examples bundle.

What still needs measuring (deferred to the on-demand evaluation, not estimated here):

- Exact token count of each instruction block via `count_tokens`, to check against the
  per-model checkpoint minimum and decide whether each prompt caches as written on the
  benchmark model (`us.anthropic.claude-sonnet-4-6`, 1,024-token minimum) and on the
  4,096-minimum models.
- Measured cache-read rate and input-token cost delta on real chunks.

The benchmark suite runs extraction on `us.anthropic.claude-sonnet-4-6`
(`integration-tests/.../benchmark_extract.py`).

## Empirical confirmation

A two-arm test confirmed the finding concretely on `us.anthropic.claude-sonnet-4-6` in
`us-west-2`. The on-demand arm is the positive control that makes the batch arm's result
meaningful — it rules out "the checkpoint was placed wrong" as the reason caching did not
apply in batch.

- **Arm A — on-demand (positive control). PASS.** Two runs. A `Converse` call with a
  `cachePoint` after a repeated large prefix returned `cacheWriteInputTokens` then
  `cacheReadInputTokens` on the second call. An `InvokeModel` call using the exact body shape
  a batch record carries (`cache_control: {type: ephemeral}` on the first content block)
  returned `cache_creation_input_tokens` then `cache_read_input_tokens` ≈ 4,240 on the repeat.
  Caching works and the marker placement is valid. (Practical note: the effective checkpoint
  minimum observed for Sonnet 4.6 was ~4,096 tokens, not the 1,024 the table lists — a
  ~1,400-token prefix did not cache; tripling it to ~4,240 did.)
- **Arm B — batch (the actual question). CONFIRMED NEGATIVE.** A 100-record batch job
  (Bedrock minimum), every record carrying the identical cached prefix and the same
  `cache_control` marker that cached in Arm A. Result: `successRecordCount: 0`,
  `errorRecordCount: 100`. Every record failed with the same error:

  > errorCode 400 (retryable: false): "You invoked an unsupported model or your request did
  > not allow prompt caching. See the documentation for more information."

The batch path does not silently drop the cache directive — it rejects the request outright.
Since the same body cached on-demand (Arm A), the batch API is the sole cause. Cost was small
(~100 short records, all errored before generation).

## Incidental observation (not part of this spike)

While reviewing `get_request_body`, the `anthropic.claude` branch built the system turn with
a malformed role literal — `{'role': 'system"', ...}` (`batch_inference_utils.py`) — instead
of using the top-level `system` field the Bedrock invoke schema expects. This was unrelated
to caching and pre-existed this work. It has since been fixed separately in #331, which moves
the system prompt to the top-level `system` field (matching the Nova branch) and adds
regression tests. Noted here for provenance only.

## References

- AWS — Prompt caching for faster model inference:
  https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
- AWS — Process multiple prompts with batch inference:
  https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html
- AWS — CreateModelInvocationJob:
  https://docs.aws.amazon.com/bedrock/latest/APIReference/API_CreateModelInvocationJob.html

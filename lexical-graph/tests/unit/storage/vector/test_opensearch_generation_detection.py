# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic AOSS generation detection in index_exists().

The resolver reads the collection generation from the AOSS control plane
(BatchGetCollection -> BatchGetCollectionGroup `generation`) and falls back to the
reactive Classic/NextGen retry when detection is unavailable. Covers:
  - NEXTGEN and CLASSIC results are each honoured
  - a collection with no group falls back (returns None)
  - AccessDenied / generic ClientError / a boto3 too old for the API all fall back
  - detection runs only for AOSS (is_sigv4_auth=True) with generation unset
  - an explicit generation override skips detection entirely
  - the endpoint host is parsed into (collection id, region)
"""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from ._opensearch_test_support import install_opensearch_mocks, FakeRequestError as _RequestError

install_opensearch_mocks()

import graphrag_toolkit.lexical_graph.storage.vector.opensearch_vector_indexes as ovi  # noqa: E402
from graphrag_toolkit.lexical_graph.config import OpenSearchServerlessGeneration  # noqa: E402


_AOSS_ENDPOINT = "https://abc123xyz.us-east-1.aoss.amazonaws.com"


@pytest.fixture(autouse=True)
def _clear_generation_cache():
    """Detection memoizes per (collection_id, region); clear it so tests stay isolated."""
    ovi._aoss_generation_cache.clear()
    yield
    ovi._aoss_generation_cache.clear()


def _client_error(code):
    return ClientError({'Error': {'Code': code, 'Message': code}}, 'BatchGetCollection')


def _fake_aoss_client(*, group_name='cg-1', generation='NEXTGEN',
                      collection_error=None, group_error=None, has_group_api=True):
    """Build a fake opensearchserverless control-plane client."""
    client = MagicMock()

    if collection_error is not None:
        client.batch_get_collection.side_effect = collection_error
    else:
        details = {'id': 'abc123xyz', 'name': 'test-collection'}
        if group_name is not None:
            details['collectionGroupName'] = group_name
        client.batch_get_collection.return_value = {'collectionDetails': [details]}

    if not has_group_api:
        del client.batch_get_collection_group  # attribute access now raises AttributeError
    elif group_error is not None:
        client.batch_get_collection_group.side_effect = group_error
    else:
        client.batch_get_collection_group.return_value = {
            'collectionGroupDetails': [{'generation': generation}]
        }
    return client


def _resolve(endpoint=_AOSS_ENDPOINT, **client_kwargs):
    """Call _resolve_aoss_generation with a mocked session client."""
    fake = _fake_aoss_client(**client_kwargs)
    with patch.object(ovi, "GraphRAGConfig") as mock_cfg:
        mock_cfg.session.client.return_value = fake
        result = ovi._resolve_aoss_generation(endpoint)
    return result, fake, mock_cfg


class TestParseAossCollection:

    def test_parses_id_and_region_from_https_host(self):
        assert ovi._parse_aoss_collection("https://abc123xyz.us-east-1.aoss.amazonaws.com") == ("abc123xyz", "us-east-1")

    def test_parses_id_and_region_without_scheme(self):
        assert ovi._parse_aoss_collection("abc123xyz.eu-west-2.aoss.amazonaws.com") == ("abc123xyz", "eu-west-2")

    def test_returns_none_for_non_aoss_host(self):
        assert ovi._parse_aoss_collection("https://my-domain.example.com") is None

    def test_returns_none_for_bare_hostname(self):
        assert ovi._parse_aoss_collection("https://endpoint") is None

    def test_returns_none_for_managed_opensearch_service_host(self):
        # Managed OpenSearch Service (es), not Serverless (aoss) -- no generation concept.
        assert ovi._parse_aoss_collection("https://search-x.us-east-1.es.amazonaws.com") is None

    def test_parses_china_partition_host(self):
        assert ovi._parse_aoss_collection(
            "https://abc123xyz.cn-north-1.aoss.amazonaws.com.cn") == ("abc123xyz", "cn-north-1")


class TestResolveGeneration:

    def test_nextgen_result_is_honoured(self):
        result, fake, _ = _resolve(generation='NEXTGEN')
        assert result is OpenSearchServerlessGeneration.NEXTGEN
        fake.batch_get_collection.assert_called_once_with(ids=['abc123xyz'])
        fake.batch_get_collection_group.assert_called_once_with(names=['cg-1'])

    def test_classic_result_is_honoured(self):
        result, _, _ = _resolve(generation='CLASSIC')
        assert result is OpenSearchServerlessGeneration.CLASSIC

    def test_no_collection_group_falls_back(self):
        result, fake, _ = _resolve(group_name=None)
        assert result is None
        fake.batch_get_collection_group.assert_not_called()

    def test_client_built_with_region_from_endpoint(self):
        _, _, mock_cfg = _resolve()
        mock_cfg.session.client.assert_called_once_with(
            'opensearchserverless', region_name='us-east-1', config=ovi._AOSS_CONTROL_PLANE_CONFIG)

    def test_result_is_memoized_per_collection(self):
        # Second lookup of the same collection is served from cache; no extra control-plane calls.
        _, fake, mock_cfg = _resolve(generation='NEXTGEN')
        assert ovi._resolve_aoss_generation(_AOSS_ENDPOINT) is OpenSearchServerlessGeneration.NEXTGEN
        mock_cfg.session.client.assert_called_once()
        fake.batch_get_collection.assert_called_once()
        fake.batch_get_collection_group.assert_called_once()

    def test_transient_failure_is_not_cached(self):
        # A failed resolution must not be cached, so a later index can retry.
        _resolve(collection_error=_client_error('ThrottlingException'))
        assert ('abc123xyz', 'us-east-1') not in ovi._aoss_generation_cache

    def test_access_denied_falls_back(self):
        result, _, _ = _resolve(collection_error=_client_error('AccessDeniedException'))
        assert result is None

    def test_generic_client_error_falls_back(self):
        result, _, _ = _resolve(collection_error=_client_error('ValidationException'))
        assert result is None

    def test_group_call_access_denied_falls_back(self):
        result, _, _ = _resolve(group_error=_client_error('AccessDeniedException'))
        assert result is None

    def test_old_boto3_without_group_api_falls_back(self):
        result, _, _ = _resolve(has_group_api=False)
        assert result is None

    def test_empty_collection_details_falls_back(self):
        fake = MagicMock()
        fake.batch_get_collection.return_value = {'collectionDetails': []}
        with patch.object(ovi, "GraphRAGConfig") as mock_cfg:
            mock_cfg.session.client.return_value = fake
            assert ovi._resolve_aoss_generation(_AOSS_ENDPOINT) is None

    def test_non_aoss_endpoint_skips_api_call(self):
        with patch.object(ovi, "GraphRAGConfig") as mock_cfg:
            assert ovi._resolve_aoss_generation("https://endpoint") is None
        mock_cfg.session.client.assert_not_called()


def _run_index_exists(generation, is_sigv4_auth=True, endpoint=_AOSS_ENDPOINT,
                      detect_kwargs=None, create_side_effects=(None,)):
    """Run index_exists() with mocked OS data client and AOSS control-plane client.
    Returns (result, os_client, aoss_client)."""
    os_client = MagicMock()
    os_client.indices.exists.return_value = False
    os_client.indices.create.side_effect = list(create_side_effects)
    aoss_client = _fake_aoss_client(**(detect_kwargs or {}))

    with patch.object(ovi, "GraphRAGConfig") as mock_cfg:
        mock_cfg.opensearch_serverless_generation = generation
        mock_cfg.opensearch_serverless_nextgen_compression = None
        mock_cfg.opensearch_engine = 'nmslib'
        mock_cfg.session.client.return_value = aoss_client
        with patch.object(ovi, "create_os_client", return_value=os_client):
            result = ovi.index_exists(endpoint, "chunk", 1024, writeable=True, is_sigv4_auth=is_sigv4_auth)
    return result, os_client, aoss_client


def _is_nextgen_body(body):
    return "method" not in body["mappings"]["properties"]["embedding"]


class TestIndexExistsUsesDetection:

    def test_detected_nextgen_uses_nextgen_mapping_directly(self):
        result, os_client, _ = _run_index_exists(generation=None, detect_kwargs={'generation': 'NEXTGEN'})
        assert result is True
        assert os_client.indices.create.call_count == 1
        assert _is_nextgen_body(os_client.indices.create.call_args.kwargs["body"])

    def test_detected_classic_uses_classic_mapping(self):
        result, os_client, _ = _run_index_exists(generation=None, detect_kwargs={'generation': 'CLASSIC'})
        assert result is True
        assert os_client.indices.create.call_count == 1
        assert not _is_nextgen_body(os_client.indices.create.call_args.kwargs["body"])

    def test_detected_classic_still_retries_if_collection_rejects(self):
        # Detection said Classic but the collection rejects it -> reactive retry still saves us.
        e = _RequestError(400, 'illegal_argument_exception',
                          {'error': {'reason': "Field parameter 'engine' is not supported"}})
        result, os_client, _ = _run_index_exists(
            generation=None, detect_kwargs={'generation': 'CLASSIC'}, create_side_effects=(e, None))
        assert result is True
        assert os_client.indices.create.call_count == 2

    def test_access_denied_falls_back_to_reactive_retry(self):
        e = _RequestError(400, 'illegal_argument_exception',
                          {'error': {'reason': "Field parameter 'engine' is not supported"}})
        result, os_client, _ = _run_index_exists(
            generation=None, detect_kwargs={'collection_error': _client_error('AccessDeniedException')},
            create_side_effects=(e, None))
        assert result is True
        assert os_client.indices.create.call_count == 2

    def test_detection_skipped_for_non_aoss_endpoint(self):
        result, os_client, aoss_client = _run_index_exists(
            generation=None, is_sigv4_auth=False)
        assert result is True
        aoss_client.batch_get_collection.assert_not_called()

    def test_detection_skipped_when_generation_explicitly_set(self):
        result, os_client, aoss_client = _run_index_exists(
            generation=OpenSearchServerlessGeneration.NEXTGEN)
        assert result is True
        aoss_client.batch_get_collection.assert_not_called()
        assert _is_nextgen_body(os_client.indices.create.call_args.kwargs["body"])

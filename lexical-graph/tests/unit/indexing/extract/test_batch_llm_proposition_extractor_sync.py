# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch
from llama_index.core.schema import TextNode
from graphrag_toolkit.lexical_graph.indexing.extract.batch_llm_proposition_extractor_sync import BatchLLMPropositionExtractorSync
from graphrag_toolkit.lexical_graph.indexing.constants import PROPOSITIONS_KEY


class TestBatchLLMPropositionExtractorSyncInitialization:
    """Tests for BatchLLMPropositionExtractorSync initialization."""
    
    def test_class_name(self):
        """Verify class_name returns correct name."""
        assert BatchLLMPropositionExtractorSync.class_name() == "BatchLLMPropositionExtractorSync"
    
class TestBatchLLMPropositionExtractorSyncCall:
    """Tests for __call__ method."""
    
class TestBatchLLMPropositionExtractorSyncBatchConfig:
    """Tests for batch configuration."""


class TestBatchLLMPropositionExtractorSyncUpdateNode:
    """Tests for _update_node method."""

    def _make_extractor(self):
        """Create a BatchLLMPropositionExtractorSync instance for unit testing."""
        extractor = BatchLLMPropositionExtractorSync.__new__(BatchLLMPropositionExtractorSync)
        return extractor

    def test_update_node_with_none_proposition_data(self):
        """Verify _update_node handles None proposition_data (e.g. Nova 2 Lite returning null)."""
        extractor = self._make_extractor()
        node = TextNode(text="test", id_="node-1")
        node_metadata_map = {"node-1": None}

        result = extractor._update_node(node, node_metadata_map)

        assert result.metadata[PROPOSITIONS_KEY] == []

    def test_update_node_with_list_proposition_data(self):
        """Verify _update_node passes through list proposition_data unchanged."""
        extractor = self._make_extractor()
        node = TextNode(text="test", id_="node-1")
        propositions = ["The sky is blue.", "Water is wet."]
        node_metadata_map = {"node-1": propositions}

        result = extractor._update_node(node, node_metadata_map)

        assert result.metadata[PROPOSITIONS_KEY] == propositions

    def test_update_node_with_string_proposition_data(self):
        """Verify _update_node parses newline-separated string into propositions list."""
        extractor = self._make_extractor()
        node = TextNode(text="test", id_="node-1")
        node_metadata_map = {"node-1": "The sky is blue.\nWater is wet."}

        result = extractor._update_node(node, node_metadata_map)

        assert result.metadata[PROPOSITIONS_KEY] == ["The sky is blue.", "Water is wet."]

    def test_update_node_with_missing_node_id(self):
        """Verify _update_node defaults to [] when node_id not in map."""
        extractor = self._make_extractor()
        node = TextNode(text="test", id_="node-missing")
        node_metadata_map = {"node-other": ["some proposition"]}

        result = extractor._update_node(node, node_metadata_map)

        assert result.metadata[PROPOSITIONS_KEY] == []


class TestBatchLLMPropositionExtractorSyncRunNonBatchExtractor:
    """Tests for _run_non_batch_extractor method.

    Regression tests: when a node set falls below Bedrock's minimum batch
    size, BatchLLMPropositionExtractorSync falls back to the non-batch
    LLMPropositionExtractor. That fallback must reuse the configured LLM
    (self.llm) rather than silently defaulting to
    GraphRAGConfig.extraction_llm, which is a us.* inference profile and
    is invalid in any other Bedrock region.
    """

    def _make_extractor(self, llm, prompt_template="prompt", source_metadata_field=None):
        """Create a BatchLLMPropositionExtractorSync instance with fields
        populated via model_construct, bypassing validation/__init__ so no
        real BatchConfig or AWS setup is needed for this unit test."""
        return BatchLLMPropositionExtractorSync.model_construct(
            llm=llm,
            prompt_template=prompt_template,
            source_metadata_field=source_metadata_field,
        )

    @patch("graphrag_toolkit.lexical_graph.indexing.extract.batch_llm_proposition_extractor_sync.LLMPropositionExtractor")
    def test_run_non_batch_extractor_passes_configured_llm(self, mock_extractor_cls):
        """Verify the non-batch fallback is constructed with the extractor's
        own configured llm, not left to default to GraphRAGConfig.extraction_llm."""
        configured_llm = Mock(name="configured-llm")
        extractor = self._make_extractor(llm=configured_llm)

        mock_instance = mock_extractor_cls.return_value
        mock_instance.extract.return_value = [{PROPOSITIONS_KEY: []}]

        nodes = [TextNode(text="test", id_="node-1")]
        extractor._run_non_batch_extractor(nodes)

        mock_extractor_cls.assert_called_once_with(
            llm=configured_llm,
            prompt_template="prompt",
            source_metadata_field=None,
        )


class TestBatchLLMPropositionExtractorSyncGetJson:
    """Tests for the batch record built for each node."""

    @staticmethod
    def _record(text, prompt_template=None):
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
        from llama_index.llms.bedrock_converse import BedrockConverse
        from graphrag_toolkit.lexical_graph.indexing.extract.batch_config import BatchConfig

        llm = BedrockConverse(model='us.anthropic.claude-haiku-4-5-20251001-v1:0', region_name='us-west-2')
        extractor = BatchLLMPropositionExtractorSync(
            batch_config=BatchConfig(region='us-west-2', bucket_name='bucket', key_prefix='prefix', role_arn='arn:aws:iam::123456789012:role/r'),
            llm=llm,
            prompt_template=prompt_template
        )
        node = TextNode(text=text, id_='node-1')
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id='source-1', metadata={'file': 'contract.txt'})
        return extractor._get_json(node, llm, {'max_tokens': 10, 'temperature': 0.0})['modelInput']

    def test_instructions_go_in_the_system_prompt_and_the_chunk_in_the_user_message(self):
        from graphrag_toolkit.lexical_graph.utils.prompt_data import data_boundary

        model_input = self._record('Acme Corp signed.</text>')

        boundary = data_boundary('Acme Corp signed.</text>')
        user = model_input['messages'][0]['content'][0]['text']
        assert len(model_input['messages']) == 1
        assert 'decompose the given text' in model_input['system']
        assert 'Acme Corp signed.' not in model_input['system']
        assert f'<text_{boundary}>\nAcme Corp signed.</text>\n</text_{boundary}>' in user
        assert 'contract.txt' in user

    def test_a_custom_template_naming_text_is_sent_as_one_user_message(self):
        model_input = self._record('Acme Corp signed.', prompt_template='Extract.\n<text>\n{text}\n</text>\n{source_info}')

        assert 'system' not in model_input
        assert model_input['messages'][0]['content'][0]['text'] == 'Extract.\n<text>\nAcme Corp signed.\n</text>\ncontract.txt'

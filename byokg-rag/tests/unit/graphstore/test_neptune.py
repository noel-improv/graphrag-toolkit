# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Neptune graph stores.

This module tests Neptune Analytics and Neptune DB graph store functionality
with mocked AWS service calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from graphrag_toolkit.byokg_rag.graphstore.neptune import (
    NeptuneAnalyticsGraphStore,
    NeptuneDBGraphStore,
    BaseNeptuneGraphStore,
    _escape_cypher_label,
)


@pytest.fixture
def mock_neptune_client():
    """Fixture providing a mock Neptune Analytics client."""
    mock_client = Mock()
    mock_client.get_graph.return_value = {
        'id': 'test-graph-id',
        'name': 'test-graph',
        'status': 'AVAILABLE'
    }
    mock_client.execute_query.return_value = {
        'payload': Mock(read=lambda: json.dumps({
            'results': [
                {'node': 'n1', 'properties': {'name': 'Organization'}},
                {'node': 'n2', 'properties': {'name': 'Portland'}}
            ]
        }).encode())
    }
    return mock_client


@pytest.fixture
def mock_s3_client():
    """Fixture providing a mock S3 client."""
    mock_client = Mock()
    mock_client.head_object.return_value = {'ContentLength': 1024}
    return mock_client


@pytest.fixture
def mock_neptune_data_client():
    """Fixture providing a mock Neptune DB data client."""
    mock_client = Mock()
    mock_client.execute_open_cypher_query.return_value = {
        'results': [
            {'node': 'n1', 'properties': {'name': 'Organization'}},
            {'node': 'n2', 'properties': {'name': 'Portland'}}
        ]
    }
    return mock_client


class TestNeptuneAnalyticsGraphStore:
    """Tests for NeptuneAnalyticsGraphStore."""
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_initialization(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify Neptune Analytics store initializes with mocked boto3."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        assert store.neptune_graph_id == 'test-graph-id'
        assert store.region == 'us-west-2'
        mock_neptune_client.get_graph.assert_called_once_with(graphIdentifier='test-graph-id')
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_get_schema(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify schema retrieval from Neptune Analytics."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        schema_response = {
            'schema': {
                'nodeLabelDetails': {
                    'Person': {'properties': ['name', 'age']},
                    'Organization': {'properties': ['name', 'industry']}
                },
                'edgeLabelDetails': {
                    'WORKS_FOR': {'properties': ['since']}
                }
            }
        }
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [schema_response]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store.get_schema()
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert 'schema' in result[0]
        mock_neptune_client.execute_query.assert_called()
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_execute_query(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify query execution with mocked responses."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        query_results = [
            {'node': 'n1', 'label': 'Person', 'name': 'John Doe'},
            {'node': 'n2', 'label': 'Organization', 'name': 'Organization'}
        ]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': query_results
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store.execute_query(
            cypher="MATCH (n:Person) RETURN n",
            parameters={}
        )
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['name'] == 'John Doe'
        assert result[1]['name'] == 'Organization'
        
        # Verify the call was made with correct parameters
        call_args = mock_neptune_client.execute_query.call_args[1]
        assert call_args['graphIdentifier'] == 'test-graph-id'
        assert call_args['queryString'] == "MATCH (n:Person) RETURN n"
        assert call_args['language'] == 'OPEN_CYPHER'
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_nodes(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify nodes() method returns node IDs."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'node': 'n1'},
                    {'node': 'n2'},
                    {'node': 'n3'}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store.nodes()
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert 'n1' in result
        assert 'n2' in result
        assert 'n3' in result
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_get_linker_tasks(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify get_linker_tasks returns expected task list."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        tasks = store.get_linker_tasks()
        
        assert isinstance(tasks, list)
        assert "entity-extraction" in tasks
        assert "path-extraction" in tasks
        assert "draft-answer-generation" in tasks
        assert "opencypher" in tasks
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    @patch.dict('os.environ', {'AWS_REGION': 'eu-west-1'})
    def test_neptune_store_region_from_env(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify region detection from AWS_REGION environment variable."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneAnalyticsGraphStore(graph_identifier='test-graph-id')
        
        assert store.region == 'eu-west-1'
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_read_from_csv_with_local_file(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify read_from_csv uploads local file and loads data."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [{'status': 'success'}]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        with patch.object(store, '_upload_to_s3') as mock_upload:
            store.read_from_csv(
                csv_file='/tmp/test.csv',
                s3_path='s3://test-bucket/data.csv',
                format='CSV'
            )
            
            mock_upload.assert_called_once_with('s3://test-bucket/data.csv', '/tmp/test.csv')
            mock_neptune_client.execute_query.assert_called()
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_read_from_csv_s3_only(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify read_from_csv loads data from existing S3 path."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [{'status': 'success'}]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        store.read_from_csv(s3_path='s3://test-bucket/existing-data.csv')
        
        mock_neptune_client.execute_query.assert_called()
        call_args = mock_neptune_client.execute_query.call_args[1]
        assert 's3://test-bucket/existing-data.csv' in call_args['queryString']
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_read_from_csv_invalid_format(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify read_from_csv raises error for invalid format."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        with pytest.raises(AssertionError, match="format must be either"):
            store.read_from_csv(s3_path='s3://test-bucket/data.csv', format='INVALID')
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_execute_query_with_parameters(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify execute_query passes parameters correctly."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [{'node': 'n1', 'name': 'Test'}]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        params = {'node_id': 'n1', 'min_age': 25}
        result = store.execute_query(
            cypher="MATCH (n) WHERE ID(n) = $node_id RETURN n",
            parameters=params
        )
        
        assert len(result) == 1
        call_args = mock_neptune_client.execute_query.call_args[1]
        assert call_args['parameters'] == params
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_get_node_text_for_embedding_grouped(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify get_node_text_for_embedding_input with grouping."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'node': 'n1', 'properties': {'name': 'John Doe', 'age': 45}},
                    {'node': 'n2', 'properties': {'name': 'John Smith', 'age': 38}}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        ids, texts = store.get_node_text_for_embedding_input(
            node_embedding_text_props={'Person': ['name', 'age']},
            group_by_node_label=True
        )
        
        assert isinstance(ids, dict)
        assert isinstance(texts, dict)
        assert 'Person' in ids
        assert 'Person' in texts
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_store_get_node_text_for_embedding_ungrouped(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify get_node_text_for_embedding_input without grouping."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'node': 'n1', 'properties': {'name': 'Organization'}},
                    {'node': 'n2', 'properties': {'name': 'DataCorp'}}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        ids, texts = store.get_node_text_for_embedding_input(
            node_embedding_text_props={'Organization': ['name']},
            group_by_node_label=False
        )
        
        assert isinstance(ids, list)
        assert isinstance(texts, list)
        assert len(ids) == 2
        assert len(texts) == 2


class TestNeptuneDBGraphStore:
    """Tests for NeptuneDBGraphStore."""
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_db_store_initialization(self, mock_session, mock_neptune_data_client, mock_s3_client):
        """Verify Neptune DB store initializes correctly."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneDBGraphStore(
            endpoint_url='https://test-cluster.us-west-2.neptune.amazonaws.com:8182',
            region='us-west-2'
        )
        
        assert store.endpoint_url == 'https://test-cluster.us-west-2.neptune.amazonaws.com:8182'
        assert store.region == 'us-west-2'
        assert store.neptune_data_client is not None
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_db_store_execute_query(self, mock_session, mock_neptune_data_client, mock_s3_client):
        """Verify Neptune DB query execution."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client
        }[service]
        
        query_results = [
            {'node': 'n1', 'name': 'Organization'},
            {'node': 'n2', 'name': 'Portland'}
        ]
        
        mock_neptune_data_client.execute_open_cypher_query.return_value = {
            'results': query_results
        }
        
        store = NeptuneDBGraphStore(
            endpoint_url='https://test-cluster.us-west-2.neptune.amazonaws.com:8182',
            region='us-west-2'
        )
        
        result = store.execute_query(
            cypher="MATCH (n) RETURN n",
            parameters={}
        )
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['name'] == 'Organization'
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_db_store_execute_query_with_parameters(self, mock_session, mock_neptune_data_client, mock_s3_client):
        """Verify Neptune DB query execution with parameters."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_data_client.execute_open_cypher_query.return_value = {
            'results': [{'node': 'n1', 'name': 'Test'}]
        }
        
        store = NeptuneDBGraphStore(
            endpoint_url='https://test-cluster.us-west-2.neptune.amazonaws.com:8182',
            region='us-west-2'
        )
        
        params = {'node_id': 'n1'}
        result = store.execute_query(
            cypher="MATCH (n) WHERE ID(n) = $node_id RETURN n",
            parameters=params
        )
        
        assert len(result) == 1
        call_args = mock_neptune_data_client.execute_open_cypher_query.call_args[1]
        assert json.loads(call_args['parameters']) == params
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_db_store_get_schema(self, mock_session, mock_neptune_data_client, mock_s3_client):
        """Verify Neptune DB get_schema retrieves and enriches schema."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_data_client.get_propertygraph_summary.return_value = {
            'payload': {
                'graphSummary': {
                    'nodeLabels': ['Person', 'Organization'],
                    'edgeLabels': ['WORKS_FOR']
                }
            }
        }
        
        # Mock execute_open_cypher_query to return different results for different queries
        # Order: triples query, edge properties query, node properties queries (Person, Organization)
        mock_neptune_data_client.execute_open_cypher_query.side_effect = [
            {'results': [{'from': ['Person'], 'edge': 'WORKS_FOR', 'to': ['Organization']}]},  # triples
            {'results': [{'props': {'since': 2020}}]},  # edge properties for WORKS_FOR
            {'results': [{'props': {'name': 'John', 'age': 30}}]},  # node properties for Person
            {'results': [{'props': {'name': 'Acme Corp', 'industry': 'Tech'}}]}  # node properties for Organization
        ]
        
        store = NeptuneDBGraphStore(
            endpoint_url='https://test-cluster.us-west-2.neptune.amazonaws.com:8182',
            region='us-west-2'
        )
        
        result = store.get_schema()
        
        assert 'nodeLabels' in result
        assert 'edgeLabels' in result
        assert 'nodeLabelDetails' in result
        assert 'edgeLabelDetails' in result
        assert 'labelTriples' in result
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_db_store_read_from_csv(self, mock_session, mock_neptune_data_client, mock_s3_client):
        """Verify Neptune DB read_from_csv starts bulk loader."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_data_client.start_loader_job.return_value = {
            'status': 'LOAD_IN_PROGRESS',
            'payload': {'loadId': 'test-load-id'}
        }
        
        store = NeptuneDBGraphStore(
            endpoint_url='https://test-cluster.us-west-2.neptune.amazonaws.com:8182',
            region='us-west-2'
        )
        
        with patch.object(store, '_upload_to_s3') as mock_upload:
            store.read_from_csv(
                csv_file='/tmp/test.csv',
                s3_path='s3://test-bucket/data.csv',
                format='CSV',
                iam_role='arn:aws:iam::123456789012:role/NeptuneLoadRole'
            )
            
            mock_upload.assert_called_once()
            mock_neptune_data_client.start_loader_job.assert_called_once()
            call_args = mock_neptune_data_client.start_loader_job.call_args[1]
            assert call_args['source'] == 's3://test-bucket/data.csv'
            assert call_args['format'] == 'csv'
            assert call_args['iamRoleArn'] == 'arn:aws:iam::123456789012:role/NeptuneLoadRole'
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_db_store_read_from_csv_invalid_format(self, mock_session, mock_neptune_data_client, mock_s3_client):
        """Verify Neptune DB read_from_csv rejects invalid formats."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneDBGraphStore(
            endpoint_url='https://test-cluster.us-west-2.neptune.amazonaws.com:8182',
            region='us-west-2'
        )
        
        with pytest.raises(AssertionError, match="format must be either"):
            store.read_from_csv(
                s3_path='s3://test-bucket/data.csv',
                format='NTRIPLES',
                iam_role='arn:aws:iam::123456789012:role/NeptuneLoadRole'
            )
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_db_store_get_node_properties(self, mock_session, mock_neptune_data_client, mock_s3_client):
        """Verify _get_node_properties enriches schema with node details."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_data_client.execute_open_cypher_query.return_value = {
            'results': [
                {'props': {'name': 'Test', 'age': 30}},
                {'props': {'name': 'Test2', 'age': 25}}
            ]
        }
        
        store = NeptuneDBGraphStore(
            endpoint_url='https://test-cluster.us-west-2.neptune.amazonaws.com:8182',
            region='us-west-2'
        )
        
        summary = {'nodeLabels': ['Person']}
        type_mapping = {'str': 'STRING', 'int': 'INTEGER'}
        
        store._get_node_properties(summary, type_mapping)
        
        assert 'nodeLabelDetails' in summary
        assert 'Person' in summary['nodeLabelDetails']
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_db_store_get_edge_properties(self, mock_session, mock_neptune_data_client, mock_s3_client):
        """Verify _get_edge_properties enriches schema with edge details."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_data_client.execute_open_cypher_query.return_value = {
            'results': [
                {'props': {'since': '2020', 'role': 'Engineer'}},
                {'props': {'since': '2021', 'role': 'Manager'}}
            ]
        }
        
        store = NeptuneDBGraphStore(
            endpoint_url='https://test-cluster.us-west-2.neptune.amazonaws.com:8182',
            region='us-west-2'
        )
        
        summary = {'edgeLabels': ['WORKS_FOR']}
        type_mapping = {'str': 'STRING', 'int': 'INTEGER'}
        
        store._get_edge_properties(summary, type_mapping)
        
        assert 'edgeLabelDetails' in summary
        assert 'WORKS_FOR' in summary['edgeLabelDetails']
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_neptune_db_store_get_triples(self, mock_session, mock_neptune_data_client, mock_s3_client):
        """Verify _get_triples enriches schema with label triples."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client
        }[service]
        
        # Mock execute_open_cypher_query to return different results for each edge label
        mock_neptune_data_client.execute_open_cypher_query.side_effect = [
            {'results': [{'from': ['Person'], 'edge': 'WORKS_FOR', 'to': ['Organization']}]},
            {'results': [{'from': ['Organization'], 'edge': 'LOCATED_IN', 'to': ['Location']}]}
        ]
        
        store = NeptuneDBGraphStore(
            endpoint_url='https://test-cluster.us-west-2.neptune.amazonaws.com:8182',
            region='us-west-2'
        )
        
        summary = {'edgeLabels': ['WORKS_FOR', 'LOCATED_IN']}
        
        store._get_triples(summary)
        
        assert 'labelTriples' in summary
        assert len(summary['labelTriples']) == 2
        assert summary['labelTriples'][0]['~type'] == 'WORKS_FOR'


class TestBaseNeptuneGraphStore:
    """Tests for BaseNeptuneGraphStore shared functionality."""
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_assign_text_repr_prop_for_nodes(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify text representation property assignment."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        store.assign_text_repr_prop_for_nodes(
            node_label_to_property_mapping={'Person': 'name', 'Organization': 'title'}
        )
        
        assert store.node_label_has_text_repr_prop('Person')
        assert store.get_text_repr_prop('Person') == 'name'
        assert store.node_label_has_text_repr_prop('Organization')
        assert store.get_text_repr_prop('Organization') == 'title'
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_assign_text_repr_prop_with_kwargs(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify text representation property assignment using kwargs."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        store.assign_text_repr_prop_for_nodes(Person='full_name', Location='city_name')
        
        assert store.node_label_has_text_repr_prop('Person')
        assert store.get_text_repr_prop('Person') == 'full_name'
        assert store.node_label_has_text_repr_prop('Location')
        assert store.get_text_repr_prop('Location') == 'city_name'
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_node_label_has_text_repr_prop_false(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify node_label_has_text_repr_prop returns False for unmapped labels."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        assert not store.node_label_has_text_repr_prop('UnknownLabel')
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_get_nodes_with_ids(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify get_nodes retrieves node details by IDs."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'node': 'n1', 'properties': {'name': 'Organization', 'industry': 'Tech'}},
                    {'node': 'n2', 'properties': {'name': 'Portland', 'country': 'USA'}}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store.get_nodes(['n1', 'n2'])
        
        assert isinstance(result, dict)
        assert 'n1' in result
        assert result['n1']['name'] == 'Organization'
        assert 'n2' in result
        assert result['n2']['name'] == 'Portland'
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_get_nodes_with_text_repr_mapping(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify get_nodes uses text representation mapping when configured."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'node': 'n1', 'properties': {'name': 'Organization', 'industry': 'Tech'}}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        store.assign_text_repr_prop_for_nodes(Organization='name')
        
        result = store.get_nodes(['n1'])
        
        assert isinstance(result, dict)
        assert 'n1' in result
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_edges(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify edges() method returns edge IDs."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'edge': 'e1'},
                    {'edge': 'e2'},
                    {'edge': 'e3'}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store.edges()
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert 'e1' in result
        assert 'e2' in result
        assert 'e3' in result
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_get_edges(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify get_edges retrieves edge details by IDs."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'edge': 'e1', 'properties': {'since': '1994', 'type': 'FOUNDED'}},
                    {'edge': 'e2', 'properties': {'since': '2010', 'type': 'WORKS_FOR'}}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store.get_edges(['e1', 'e2'])
        
        assert isinstance(result, dict)
        assert 'e1' in result
        assert result['e1']['since'] == '1994'
        assert 'e2' in result
        assert result['e2']['since'] == '2010'
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_get_one_hop_edges_without_triplets(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify get_one_hop_edges returns edge IDs when return_triplets=False."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'node': 'n1', 'edge': 'e1', 'edge_type': 'FOUNDED', 'dst_node': 'n2'}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store.get_one_hop_edges(['n1'], return_triplets=False)
        
        assert isinstance(result, dict)
        assert 'n1' in result
        assert 'FOUNDED' in result['n1']
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_get_one_hop_edges_with_triplets(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify get_one_hop_edges returns triplets when return_triplets=True."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'node': 'n1', 'edge': 'e1', 'edge_type': 'FOUNDED', 'dst_node': 'n2'}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store.get_one_hop_edges(['n1'], return_triplets=True)
        
        assert isinstance(result, dict)
        assert 'n1' in result
        assert 'FOUNDED' in result['n1']
        triplets = list(result['n1']['FOUNDED'])
        assert len(triplets) > 0
        assert triplets[0][0] == 'n1'
        assert triplets[0][1] == 'FOUNDED'
        assert triplets[0][2] == 'n2'
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_nodes_with_node_type(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify nodes() method filters by node type."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'node': 'n1'},
                    {'node': 'n2'}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store.nodes(node_type='Person')
        
        assert isinstance(result, list)
        assert len(result) == 2
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_nodes_with_text_repr_properties(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify nodes() returns text representation when configured."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_neptune_client.execute_query.return_value = {
            'payload': Mock(read=lambda: json.dumps({
                'results': [
                    {'node': 'n1', 'properties': {'name': 'John Doe'}, 'node_labels': ['Person']},
                    {'node': 'n2', 'properties': {'name': 'Organization'}, 'node_labels': ['Organization']}
                ]
            }).encode())
        }
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        store.assign_text_repr_prop_for_nodes(Person='name', Organization='name')
        
        result = store.nodes()
        
        assert isinstance(result, list)
        assert 'John Doe' in result
        assert 'Organization' in result
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_s3_file_exists_true(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify _s3_file_exists returns True when file exists."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        mock_s3_client.head_object.return_value = {'ContentLength': 1024}
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store._s3_file_exists('s3://test-bucket/test-file.csv')
        
        assert result is True
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_s3_file_exists_false(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify _s3_file_exists returns False when file doesn't exist."""
        from botocore.exceptions import ClientError
        
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        error_response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
        mock_s3_client.head_object.side_effect = ClientError(error_response, 'HeadObject')
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store._s3_file_exists('s3://test-bucket/missing-file.csv')
        
        assert result is False
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_s3_file_exists_none_path(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify _s3_file_exists returns False for None path."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        result = store._s3_file_exists(None)
        
        assert result is False
    
    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_upload_to_s3_with_file_contents(self, mock_session, mock_neptune_client, mock_s3_client):
        """Verify _upload_to_s3 uploads file contents."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptune-graph': mock_neptune_client,
            's3': mock_s3_client
        }[service]
        
        store = NeptuneAnalyticsGraphStore(
            graph_identifier='test-graph-id',
            region='us-west-2'
        )
        
        store._upload_to_s3('s3://test-bucket/test.csv', file_contents='test,data\n1,2')

        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args[1]
        assert call_args['Bucket'] == 'test-bucket'
        assert call_args['Body'] == 'test,data\n1,2'


# Labels read from the property graph summary are interpolated into
# backtick-quoted positions in schema-discovery Cypher. A label containing a
# backtick would otherwise break out of the identifier and inject Cypher.
# Cypher's escape rule for backtick-quoted identifiers is to double the
# backtick.

class TestEscapeCypherLabel:
    """Unit tests for the label-escaping helper."""

    def test_plain_label_unchanged(self):
        assert _escape_cypher_label('Person') == 'Person'

    def test_empty_string_unchanged(self):
        assert _escape_cypher_label('') == ''

    @pytest.mark.parametrize('raw,escaped', [
        pytest.param('a`b', 'a``b', id='backtick-in-middle'),
        pytest.param('`evil', '``evil', id='backtick-at-start'),
        pytest.param('evil`', 'evil``', id='backtick-at-end'),
        pytest.param('a``b', 'a````b', id='already-doubled'),
        pytest.param('`) DETACH DELETE n //', '``) DETACH DELETE n //',
                     id='full-cypher-breakout'),
        pytest.param('```', '``````', id='triple-backtick'),
    ])
    def test_backticks_are_doubled(self, raw, escaped):
        assert _escape_cypher_label(raw) == escaped


def _captured_queries(mock_client):
    """Return every cypher string passed to execute_open_cypher_query."""
    return [
        call.kwargs.get('openCypherQuery') or call.args[0]
        for call in mock_client.execute_open_cypher_query.call_args_list
    ]


# Payload chosen to break out of backtick-quoted label position and append a
# destructive clause if the escape helper does not fire. The prefix "evil"
# keeps the escape signature ("evil``") distinct from the raw breakout
# ("evil`") even after the surrounding template backticks concatenate.
_BREAKOUT_LABEL = 'evil`) MATCH (x) DETACH DELETE x //'
_RAW_BREAKOUT_FRAGMENT = 'evil`) MATCH'
_ESCAPED_BREAKOUT_FRAGMENT = 'evil``) MATCH'


class TestSchemaDiscoveryEscapesLabels:
    """Each schema-discovery method must escape backticks in dynamic labels
    before sending Cypher to Neptune. Covers all three sinks."""

    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def _store(self, mock_session, mock_neptune_data_client, mock_s3_client):
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client,
        }[service]
        return NeptuneDBGraphStore(
            endpoint_url='https://example.us-east-1.neptune.amazonaws.com:8182',
            region='us-east-1',
        )

    def test_get_edge_properties_escapes_backtick(
        self, mock_neptune_data_client, mock_s3_client,
    ):
        mock_neptune_data_client.execute_open_cypher_query.return_value = {
            'results': [],
        }
        store = self._store(
            mock_neptune_data_client=mock_neptune_data_client,
            mock_s3_client=mock_s3_client,
        )
        summary = {'edgeLabels': [_BREAKOUT_LABEL]}

        store._get_edge_properties(summary, {'str': 'STRING'})

        sent = _captured_queries(mock_neptune_data_client)
        assert len(sent) == 1
        assert _ESCAPED_BREAKOUT_FRAGMENT in sent[0]
        assert _RAW_BREAKOUT_FRAGMENT not in sent[0]
        # Raw label is preserved in the Python-side dict key, not the cypher.
        assert _BREAKOUT_LABEL in summary['edgeLabelDetails']

    def test_get_node_properties_escapes_backtick(
        self, mock_neptune_data_client, mock_s3_client,
    ):
        mock_neptune_data_client.execute_open_cypher_query.return_value = {
            'results': [],
        }
        store = self._store(
            mock_neptune_data_client=mock_neptune_data_client,
            mock_s3_client=mock_s3_client,
        )
        summary = {'nodeLabels': [_BREAKOUT_LABEL]}

        store._get_node_properties(summary, {'str': 'STRING'})

        sent = _captured_queries(mock_neptune_data_client)
        assert len(sent) == 1
        assert _ESCAPED_BREAKOUT_FRAGMENT in sent[0]
        assert _RAW_BREAKOUT_FRAGMENT not in sent[0]
        assert _BREAKOUT_LABEL in summary['nodeLabelDetails']

    def test_get_triples_escapes_backtick(
        self, mock_neptune_data_client, mock_s3_client,
    ):
        mock_neptune_data_client.execute_open_cypher_query.return_value = {
            'results': [],
        }
        store = self._store(
            mock_neptune_data_client=mock_neptune_data_client,
            mock_s3_client=mock_s3_client,
        )
        summary = {'edgeLabels': [_BREAKOUT_LABEL]}

        store._get_triples(summary)

        sent = _captured_queries(mock_neptune_data_client)
        assert len(sent) == 1
        assert _ESCAPED_BREAKOUT_FRAGMENT in sent[0]
        assert _RAW_BREAKOUT_FRAGMENT not in sent[0]

    def test_plain_label_flows_unchanged(
        self, mock_neptune_data_client, mock_s3_client,
    ):
        """Positive path: a canonical label reaches the cypher in
        backtick-quoted form without modification."""
        mock_neptune_data_client.execute_open_cypher_query.return_value = {
            'results': [],
        }
        store = self._store(
            mock_neptune_data_client=mock_neptune_data_client,
            mock_s3_client=mock_s3_client,
        )
        summary = {'nodeLabels': ['Person']}

        store._get_node_properties(summary, {'str': 'STRING'})

        sent = _captured_queries(mock_neptune_data_client)
        assert '`Person`' in sent[0]


class TestSchemaDiscoveryIndirectCaller:
    """get_schema -> _refresh_schema dispatches into all three sinks. The
    guard must fire on the public entry point, not just the helpers."""

    @patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
    def test_get_schema_escapes_backtick_in_every_sink(
        self, mock_session, mock_neptune_data_client, mock_s3_client,
    ):
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.client.side_effect = lambda service, **kwargs: {
            'neptunedata': mock_neptune_data_client,
            's3': mock_s3_client,
        }[service]

        mock_neptune_data_client.get_propertygraph_summary.return_value = {
            'payload': {
                'graphSummary': {
                    'nodeLabels': [_BREAKOUT_LABEL],
                    'edgeLabels': [_BREAKOUT_LABEL],
                },
            },
        }
        mock_neptune_data_client.execute_open_cypher_query.return_value = {
            'results': [],
        }

        store = NeptuneDBGraphStore(
            endpoint_url='https://example.us-east-1.neptune.amazonaws.com:8182',
            region='us-east-1',
        )

        store.get_schema()

        sent = _captured_queries(mock_neptune_data_client)
        # _refresh_schema fans out to triples + node_properties + edge_properties.
        assert len(sent) >= 3
        for cypher in sent:
            assert _ESCAPED_BREAKOUT_FRAGMENT in cypher
            assert _RAW_BREAKOUT_FRAGMENT not in cypher


# Red-state proof: with the escape helper patched to identity, the breakout
# payload reaches execute_open_cypher_query unescaped. Pins the mitigation
# to the helper and catches a future refactor that drops the call.
@patch(
    'graphrag_toolkit.byokg_rag.graphstore.neptune._escape_cypher_label',
    side_effect=lambda label: label,
)
@patch('graphrag_toolkit.byokg_rag.graphstore.neptune.boto3.Session')
def test_payload_reaches_cypher_when_escape_disabled(
    mock_session, mock_escape, mock_neptune_data_client, mock_s3_client,
):
    mock_session_instance = Mock()
    mock_session.return_value = mock_session_instance
    mock_session_instance.client.side_effect = lambda service, **kwargs: {
        'neptunedata': mock_neptune_data_client,
        's3': mock_s3_client,
    }[service]
    mock_neptune_data_client.execute_open_cypher_query.return_value = {
        'results': [],
    }

    store = NeptuneDBGraphStore(
        endpoint_url='https://example.us-east-1.neptune.amazonaws.com:8182',
        region='us-east-1',
    )
    summary = {'edgeLabels': [_BREAKOUT_LABEL]}

    store._get_edge_properties(summary, {'str': 'STRING'})

    sent = _captured_queries(mock_neptune_data_client)
    assert _RAW_BREAKOUT_FRAGMENT in sent[0]
    assert _ESCAPED_BREAKOUT_FRAGMENT not in sent[0]
    assert 'DETACH DELETE x' in sent[0]

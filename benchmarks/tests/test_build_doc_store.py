# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

from benchmarks.scripts.benchmark_build import _create_doc_store, _latest_s3_collection


S3_ENV = {
    'BENCHMARK_DOC_STORE': 's3',
    'AWS_REGION_NAME': 'us-west-2',
    'AWS_DEFAULT_REGION': 'us-west-2',
    'S3_RESULTS_BUCKET': 'test-bucket',
    'S3_RESULTS_PREFIX': 'test-prefix',
}


def _paginator_returning(collection_ids):
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {'CommonPrefixes': [
            {'Prefix': f'test-prefix/doc-store/wikihow/{c}/'} for c in collection_ids
        ]}
    ]
    return paginator


class TestLatestS3Collection(unittest.TestCase):

    def _run(self, collection_ids):
        s3 = MagicMock()
        s3.get_paginator.return_value = _paginator_returning(collection_ids)
        with patch('benchmarks.scripts.benchmark_build.GraphRAGConfig') as config:
            config.s3 = s3
            return _latest_s3_collection('test-bucket', 'test-prefix/doc-store/wikihow')

    def test_picks_the_newest_collection(self):
        """Collection ids are timestamps, so lexical order is time order."""
        result = self._run(['20260812-024153', '20260812-101259', '20260812-034414'])

        self.assertEqual(result, '20260812-101259')

    def test_single_collection(self):
        self.assertEqual(self._run(['20260812-024153']), '20260812-024153')

    def test_no_collections_raises_rather_than_building_nothing(self):
        with self.assertRaises(ValueError) as ctx:
            self._run([])

        self.assertIn('No collections found', str(ctx.exception))


class TestCreateDocStore(unittest.TestCase):

    def test_file_doc_store_is_the_default(self):
        with patch.dict('os.environ', {}, clear=True):
            with patch('benchmarks.scripts.benchmark_build.FileBasedDocs') as file_docs:
                _create_doc_store('wikihow', 'source-data', {})

        file_docs.assert_called_once()
        self.assertEqual(
            file_docs.call_args.kwargs['docs_directory'], 'source-data/wikihow/extracted'
        )

    def test_s3_doc_store_selected_by_env(self):
        s3 = MagicMock()
        s3.get_paginator.return_value = _paginator_returning(['20260812-024153'])

        with patch.dict('os.environ', S3_ENV, clear=True):
            with (
                patch('benchmarks.scripts.benchmark_build.GraphRAGConfig') as config,
                patch('benchmarks.scripts.benchmark_build.S3BasedDocs') as s3_docs,
            ):
                config.s3 = s3
                _create_doc_store('wikihow', 'source-data', {})

        kwargs = s3_docs.call_args.kwargs
        self.assertEqual(kwargs['bucket_name'], 'test-bucket')
        self.assertEqual(kwargs['key_prefix'], 'test-prefix/doc-store/wikihow')
        self.assertEqual(kwargs['collection_id'], '20260812-024153')

    def test_explicit_collection_id_wins_over_latest(self):
        """A sweep leaves several collections; the run must be able to name one."""
        s3 = MagicMock()
        s3.get_paginator.return_value = _paginator_returning(['20260812-999999'])

        env = dict(S3_ENV, BENCHMARK_COLLECTION_ID='20260812-024153')
        with patch.dict('os.environ', env, clear=True):
            with (
                patch('benchmarks.scripts.benchmark_build.GraphRAGConfig') as config,
                patch('benchmarks.scripts.benchmark_build.S3BasedDocs') as s3_docs,
            ):
                config.s3 = s3
                _create_doc_store('wikihow', 'source-data', {})

        self.assertEqual(s3_docs.call_args.kwargs['collection_id'], '20260812-024153')

    def test_missing_s3_vars_fail_before_any_call(self):
        env = {'BENCHMARK_DOC_STORE': 's3', 'AWS_REGION_NAME': 'us-west-2'}

        with patch.dict('os.environ', env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                _create_doc_store('wikihow', 'source-data', {})

        message = str(ctx.exception)
        self.assertIn('S3_RESULTS_BUCKET', message)
        self.assertIn('S3_RESULTS_PREFIX', message)


if __name__ == '__main__':
    unittest.main()

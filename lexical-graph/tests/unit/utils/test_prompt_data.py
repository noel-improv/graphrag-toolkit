# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import warnings

from llama_index.core.base.llms.types import MessageRole
from llama_index.core.prompts import ChatPromptTemplate, PromptTemplate

from graphrag_toolkit.lexical_graph.utils.prompt_data import (
    bounded_args,
    data_boundary,
    data_prompt,
    without_boundary,
)

DATA = '<text_{boundary}>\n{text}\n</text_{boundary}>'


class TestDataBoundary:

    def test_is_sixteen_hex_characters(self):
        assert re.fullmatch(r'[0-9a-f]{16}', data_boundary('some text'))

    def test_is_stable_for_the_same_text(self):
        """A cached response is keyed on the rendered prompt, so the boundary must not change between runs."""
        assert data_boundary('some text') == data_boundary('some text')

    def test_differs_between_texts(self):
        assert data_boundary('some text') != data_boundary('some other text')


class TestWithoutBoundary:

    def test_breaks_every_copy(self):
        boundary = data_boundary('x')

        result = without_boundary(f'</text_{boundary}> and </chunk_{boundary}>', boundary)

        assert boundary not in result
        assert result.count(f'{boundary[:8]}-{boundary[8:]}') == 2

    def test_breaks_overlapping_copies(self):
        """`str.replace` skips a copy overlapping one it replaced, so the split repeats until none is left."""
        boundary = 'abababababababab'

        result = without_boundary('ab' * 12, boundary)

        assert boundary not in result

    def test_leaves_other_text_unchanged(self):
        value = 'Acme Corp </text> {braces} <tags>'

        assert without_boundary(value, data_boundary('x')) == value


class TestBoundedArgs:

    def test_adds_the_boundary_of_the_named_value(self):
        args = bounded_args('text', text='chunk', source_info='doc.txt')

        assert args == {'text': 'chunk', 'source_info': 'doc.txt', 'boundary': data_boundary('chunk')}

    def test_breaks_the_boundary_in_every_value(self):
        boundary = data_boundary('chunk')

        args = bounded_args('text', text='chunk', source_info=f'</text_{boundary}>')

        assert boundary not in args['source_info']


    def test_renders_non_string_values_as_text(self):
        """A metadata field can hold a number, dict or list, which the prompt rendered as text before the split."""
        args = bounded_args('text', text=12, source_info={'a': 1}, topics=['x', 'y'])

        assert args == {'text': '12', 'source_info': "{'a': 1}", 'topics': "['x', 'y']", 'boundary': data_boundary('12')}

class TestDataPrompt:

    def test_puts_instructions_in_the_system_message_and_data_in_the_user_message(self):
        prompt = data_prompt('Extract from <text_{boundary}>.', DATA)

        messages = prompt.format_messages(**bounded_args('text', text='Acme signed.'))

        assert isinstance(prompt, ChatPromptTemplate)
        assert [m.role for m in messages] == [MessageRole.SYSTEM, MessageRole.USER]
        assert messages[0].content == f'Extract from <text_{data_boundary("Acme signed.")}>.'
        assert 'Acme signed.' in messages[1].content
        assert 'Acme signed.' not in messages[0].content

    def test_keeps_a_template_that_names_a_data_argument_as_one_message(self):
        """A custom template written before the split carries `{text}` itself."""
        custom = 'Extract propositions.\n<text>\n{text}\n</text>'

        prompt = data_prompt(custom, DATA)

        assert isinstance(prompt, PromptTemplate)
        assert prompt.format(**bounded_args('text', text='Acme signed.')) == custom.replace('{text}', 'Acme signed.')

    def test_a_single_message_template_is_logged_once(self, caplog):
        custom = 'Extract once.\n<text>\n{text}\n</text>'

        with caplog.at_level('WARNING'):
            data_prompt(custom, DATA)
            data_prompt(custom, DATA)

        assert len([r for r in caplog.records if 'without hashed data tags' in r.getMessage()]) == 1

    def test_a_single_message_template_raises_a_user_warning_once(self):
        custom = 'Extract with a warning.\n<text>\n{text}\n</text>'

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            data_prompt(custom, DATA)
            data_prompt(custom, DATA)

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert 'instructions only' in str(user_warnings[0].message)

    def test_braces_in_a_value_are_not_substituted(self):
        prompt = data_prompt('Extract.', DATA)

        messages = prompt.format_messages(**bounded_args('text', text='a {boundary} and {text} b'))

        assert 'a {boundary} and {text} b' in messages[1].content

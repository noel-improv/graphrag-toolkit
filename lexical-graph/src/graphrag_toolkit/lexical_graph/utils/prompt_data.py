# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from functools import lru_cache
from hashlib import sha256
from typing import Any, Dict, Union

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate, PromptTemplate

logger = logging.getLogger(__name__)

BOUNDARY_ARG = 'boundary'

_FORMAT_FIELD = re.compile(r'\{([A-Za-z_][A-Za-z0-9_]*)\}')

def data_boundary(text:str) -> str:
    """The suffix for the tags around untrusted data: the first 16 hex characters of the text's SHA-256.

    The same text always gets the same suffix, so cached responses still match,
    and a document cannot close its tag without containing its own hash.
    """
    return sha256(text.encode('utf-8')).hexdigest()[:16]

def without_boundary(value:str, boundary:str) -> str:
    """`value` with every copy of `boundary` split by a hyphen, so it cannot close a tag carrying it."""
    while boundary in value:
        value = value.replace(boundary, f'{boundary[:8]}-{boundary[8:]}')
    return value

def bounded_args(boundary_from:str, **values:Any) -> Dict[str, str]:
    """Prompt arguments for `values` as strings, plus the boundary derived from the value named `boundary_from`."""
    texts = {name: str(value) for (name, value) in values.items()}
    boundary = data_boundary(texts[boundary_from])
    args = {name: without_boundary(text, boundary) for (name, text) in texts.items()}
    args[BOUNDARY_ARG] = boundary
    return args

@lru_cache(maxsize=None)
def _warn_single_message(instructions:str) -> None:
    logger.warning('A prompt template names its data arguments itself, so it is sent as one message without hashed data tags')

def data_prompt(instructions:str, data_template:str) -> Union[PromptTemplate, ChatPromptTemplate]:
    """`instructions` as the system message and `data_template` as the user message.

    A template that names one of the data arguments itself is a single-message
    template written before the split, and is rendered as one user message, as before.
    """
    data_args = set(_FORMAT_FIELD.findall(data_template)) - {BOUNDARY_ARG}
    if any(f'{{{arg}}}' in instructions for arg in data_args):
        _warn_single_message(instructions)
        return PromptTemplate(instructions)
    return ChatPromptTemplate(message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=instructions),
        ChatMessage(role=MessageRole.USER, content=data_template),
    ])

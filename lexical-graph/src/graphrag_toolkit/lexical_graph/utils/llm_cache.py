# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from botocore.config import Config
from hashlib import sha256
from typing import Optional, Any, Union

from graphrag_toolkit.lexical_graph import ModelError
from graphrag_toolkit.lexical_graph.utils.bedrock_utils import *
from graphrag_toolkit.lexical_graph.config import GraphRAGConfig, BOTOCORE_DEFAULT_MAX_POOL_CONNECTIONS

from llama_index.core.llms.llm import LLM
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.types import TokenGen


logger = logging.getLogger(__name__) 

c_red, c_blue, c_green, c_cyan, c_norm = "\x1b[31m",'\033[94m','\033[92m', '\033[96m', '\033[0m'

MAX_ATTEMPTS = 2
TIMEOUT = 60.0


def bedrock_client_config() -> Config:
    """
    Build the botocore config for the bedrock-runtime client.

    Extraction issues one concurrent request per thread, but this client was
    created without a pool size, so it took botocore's default of 10. Once the
    pool is exhausted botocore discards and reopens connections, paying a fresh
    TCP and TLS handshake per call, which costs back the concurrency the extra
    threads were meant to buy.

    Measured over a 500-document sweep: achieved concurrency sat at ~16 in-flight
    requests whether 32, 64 or 128 threads were configured, while Bedrock's own
    InvocationLatency stayed flat at ~6.6s - so whatever capped it was on this
    side of the wire.

    GraphRAGConfig sizes the S3 pool the same way in `_client_config`, added by
    PR #420; this brings the Bedrock client into line with it.
    """
    num_threads = GraphRAGConfig.extraction_num_threads_per_worker

    # Sizing the pool must never be the reason a client cannot be built, so
    # anything that is not a usable thread count falls back to the default
    # rather than raising out of client creation.
    if not isinstance(num_threads, int) or num_threads < 1:
        logger.debug(
            f'Unusable thread count for pool sizing, using the botocore default '
            f'[value: {num_threads!r}]'
        )
        num_threads = BOTOCORE_DEFAULT_MAX_POOL_CONNECTIONS

    return Config(
        retries={'max_attempts': MAX_ATTEMPTS, 'mode': 'standard'},
        connect_timeout=TIMEOUT,
        read_timeout=TIMEOUT,
        max_pool_connections=max(BOTOCORE_DEFAULT_MAX_POOL_CONNECTIONS, num_threads),
    )


class LLMCache(BaseModel):

    llm:LLM = Field(description='LLM whose responses may be cached')
    enable_cache:Optional[bool] = Field(description='Whether the cache is enabled or disabled', default=False)
    verbose_prompt:Optional[bool] = Field(default=False)
    verbose_response:Optional[bool] = Field(default=False)

    def stream(
         self,
        prompt: BasePromptTemplate,
        **prompt_args: Any
    ) -> TokenGen:
        response = None

        if self.verbose_prompt:
            logger.info('%s%s%s', c_blue, prompt.format(**prompt_args), c_norm)

        try:
            if isinstance(self.llm, BedrockConverse):
                if not hasattr(self.llm, '_client'):
                    config = bedrock_client_config()
                    
                    session = GraphRAGConfig.session
                    self.llm._client = session.client('bedrock-runtime', config=config, region_name=self.llm.region_name)
            response = self.llm.stream(prompt, **prompt_args)
        except Exception as e:
            raise ModelError(f'{e!s} [Model config: {self.llm.to_json()}]') from e
            
        return response

    def predict(
        self,
        prompt: BasePromptTemplate,
        **prompt_args: Any
    ) -> str:
        """
        Predicts a response based on the given prompt and dynamic arguments using the configured
        language model (LLM). Supports caching of responses to enhance efficiency for repeated
        queries and handles verbose logging for debugging or monitoring purposes.

        The function dynamically adapts caching behavior depending on the configuration. If caching
        is disabled, responses are generated directly using the LLM. If caching is enabled, it calculates
        a unique cache key based on the prompt and LLM configuration, then fetches responses from the
        cache, if available, or generates and stores them for future use.

        The function ensures proper handling of potential errors during model execution and writes
        extensive logs when verbosity options are enabled, aiding in thorough tracking during execution.

        Args:
            prompt: A pre-formatted BasePromptTemplate instance containing the template definition
                to generate the LLM response.
            **prompt_args: Arbitrary keyword arguments that provide dynamic content to fill
                in the placeholders of the given prompt template.

        Returns:
            str: The generated or cached response from the LLM.

        Raises:
            ModelError: If there is any exception while interacting with the LLM, detailed
                configuration information is included to aid debugging.
        """
        response = None

        if self.verbose_prompt:
            logger.info('%s%s%s', c_blue, prompt.format(**prompt_args), c_norm)

        if not self.enable_cache:
            try:
                if isinstance(self.llm, BedrockConverse):
                    if not hasattr(self.llm, '_client'):
                        config = bedrock_client_config()
                        
                        session = GraphRAGConfig.session
                        self.llm._client = session.client('bedrock-runtime', config=config, region_name=self.llm.region_name)
                response = self.llm.predict(prompt, **prompt_args)
            except Exception as e:
                raise ModelError(f'{e!s} [Model config: {self.llm.to_json()}]') from e
        else:

            prompt_args_copy = prompt_args.copy()
            for key in prompt_args.get('exclude_cache_keys', []):
                del prompt_args_copy[key]

            cache_key = f'{self.llm.to_json()},{prompt.format(**prompt_args_copy)}'
            cache_hex = sha256(cache_key.encode('utf-8')).hexdigest()
            cache_file = f'cache/llm/{cache_hex}.txt'

            if os.path.exists(cache_file):
                logger.debug('%sCached response %s%s', c_blue, cache_file, c_norm)
                with open(cache_file, 'r', encoding='utf-8') as f:
                    response = f.read()
            else:
                try:
                    if isinstance(self.llm, BedrockConverse):
                        if not hasattr(self.llm, '_client'):
                            config = bedrock_client_config()
                            
                            session = GraphRAGConfig.session
                            self.llm._client = session.client('bedrock-runtime', config=config, region_name=self.llm.region_name)
                    response = self.llm.predict(prompt, **prompt_args)
                except Exception as e:
                    raise ModelError(f'{e!s} Model config: {self.llm.to_json()}') from e
                os.makedirs(os.path.dirname(os.path.realpath(cache_file)), exist_ok=True)
                with open(cache_file, 'w') as f:
                    f.write(response)

        if self.verbose_response:
            logger.info('%s%s%s', c_green, response, c_norm)
            
        return response
    
    @property
    def model(self):
        if not isinstance(self.llm, BedrockConverse):
            raise ModelError(f'Invalid LLM type: {type(self.llm)} does not support model')
        return self.llm.model

    
LLMCacheType = Union[LLM, LLMCache]
    



    
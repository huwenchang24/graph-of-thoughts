# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import backoff
import os
import random
import time
import logging
import asyncio
from typing import List, Dict, Union
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion
from contextlib import asynccontextmanager

from .abstract_language_model import AbstractLanguageModel

logger = logging.getLogger(__name__)

class DeepSeek(AbstractLanguageModel):
    """
    The DeepSeek class handles interactions with the DeepSeek models using the provided configuration.
    It uses the OpenAI-compatible API interface provided by DeepSeek.

    Inherits from the AbstractLanguageModel and implements its abstract methods.
    """

    def __init__(
        self, config_path: str = "", model_name: str = "deepseek", cache: bool = False
    ) -> None:
        """
        Initialize the DeepSeek instance with configuration, model details, and caching options.

        :param config_path: Path to the configuration file. Defaults to "".
        :type config_path: str
        :param model_name: Name of the model, default is 'deepseek'. Used to select the correct configuration.
        :type model_name: str
        :param cache: Flag to determine whether to cache responses. Defaults to False.
        :type cache: bool
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        # The model_id is the id of the model that is used for deepseek
        self.model_id: str = self.config["model_id"]
        # The prompt_token_cost and response_token_cost are the costs for 1000 prompt tokens and 1000 response tokens respectively.
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        # The temperature of a model is defined as the randomness of the model's output.
        self.temperature: float = self.config["temperature"]
        # The maximum number of tokens to generate in the chat completion.
        self.max_tokens: int = self.config["max_tokens"]
        # The stop sequence is a sequence of tokens that the model will stop generating at.
        self.stop: Union[str, List[str]] = self.config["stop"]
        # The API base URL for DeepSeek
        self.api_base: str = self.config.get("api_base", "https://api.deepseek.com/v1")
        # Get API key
        self.api_key: str = os.getenv("DEEPSEEK_API_KEY", self.config["api_key"])
        if self.api_key == "":
            raise ValueError("DEEPSEEK_API_KEY is not set")
        # Maximum concurrent requests
        self.max_concurrent = 10
        # Event loop
        self._loop = None
        # Add a class-level client
        self._client = None
        self._client_lock = asyncio.Lock()

    @asynccontextmanager
    async def _get_client(self):
        """
        Optimized client management, using singleton pattern
        """
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        try:
            yield self._client
        except Exception:
            # Only close and recreate client when there's an exception
            if self._client:
                await self._client.close()
                self._client = None
            raise

    def _ensure_loop(self):
        """
        Ensure we have a running event loop.
        """
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def query(
        self, query: str, num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        """
        Query the DeepSeek model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the DeepSeek model.
        :rtype: Dict
        """
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        # Ensure we have a running event loop
        self._ensure_loop()

        # Run async query in event loop
        try:
            responses = self._loop.run_until_complete(self._async_query(query, num_responses))
        except RuntimeError as e:
            if str(e) == 'Event loop is closed':
                # If the loop is closed, create a new one
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                responses = self._loop.run_until_complete(self._async_query(query, num_responses))
            else:
                raise

        if self.cache:
            self.response_cache[query] = responses
        return responses[0] if len(responses) == 1 else responses

    async def _async_query(
        self, query: str, num_responses: int = 1
    ) -> List[ChatCompletion]:
        """
        Asynchronously query the DeepSeek model for multiple responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: List of responses from the DeepSeek model.
        :rtype: List[ChatCompletion]
        """
        messages = [{"role": "user", "content": query}]
        tasks = []
        responses = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with self._get_client() as client:
            # Create tasks for all required responses
            for _ in range(num_responses):
                task = asyncio.create_task(self._bounded_chat(messages, semaphore, client))
                tasks.append(task)
            
            # Wait for all tasks to complete
            completed_responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses and handle any errors
            for response in completed_responses:
                if isinstance(response, Exception):
                    self.logger.warning(f"Error in deepseek: {response}")
                    continue
                responses.append(response)
            
            if not responses:
                raise Exception("All requests failed")
            
            return responses

    async def _bounded_chat(
        self, messages: List[Dict], semaphore: asyncio.Semaphore, client: AsyncOpenAI
    ) -> ChatCompletion:
        """
        Execute chat completion with a semaphore to bound concurrent requests.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param semaphore: Semaphore to limit concurrent requests
        :type semaphore: asyncio.Semaphore
        :param client: AsyncOpenAI client instance
        :type client: AsyncOpenAI
        :return: The DeepSeek model's response.
        :rtype: ChatCompletion
        """
        async with semaphore:
            return await self.chat(messages, client)

    @backoff.on_exception(
        backoff.expo,
        OpenAIError,
        # Reduce maximum retry time and number of tries
        max_time=5,
        max_tries=3
    )
    async def chat(self, messages: List[Dict], client: AsyncOpenAI) -> ChatCompletion:
        """
        Send chat messages to the DeepSeek model and retrieves the model's response.
        Implements backoff on API error.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param client: AsyncOpenAI client instance
        :type client: AsyncOpenAI
        :return: The DeepSeek model's response.
        :rtype: ChatCompletion
        """
        response = await client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,  # DeepSeek API doesn't support n>1
            stop=self.stop,
        )

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )
        self.logger.info(
            f"This is the response from deepseek: {response}"
            f"\nThis is the cost of the response: {self.cost}"
        )
        return response

    def get_response_texts(
        self, query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the DeepSeek model.
        :type query_response: Union[List[ChatCompletion], ChatCompletion]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, List):
            query_response = [query_response]
        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]
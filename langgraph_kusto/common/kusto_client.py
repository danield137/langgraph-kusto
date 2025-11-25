from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    EnvironmentCredential,
    InteractiveBrowserCredential,
    ManagedIdentityCredential,
    SharedTokenCacheCredential,
    TokenCachePersistenceOptions,
)
from azure.kusto.data import ClientRequestProperties
from azure.kusto.data import KustoClient as ADXKustoClient
from azure.kusto.data import KustoConnectionStringBuilder

from langgraph_kusto.common import KustoConfig


class KustoClient:
    _client_cache: dict[str, ADXKustoClient] = {}
    _client: ADXKustoClient

    def __init__(self, *, config: KustoConfig) -> None:
        self._config = config

        cached = self._client_cache.get(config.cluster_uri)
        if cached is not None:
            self._client = cached
            return

        # Enable token caching for local development
        # allow_unencrypted_storage=True allows fallback to plaintext cache if encryption fails
        cache_options = TokenCachePersistenceOptions(allow_unencrypted_storage=True)

        # Use a chained credential:
        # 1. SharedTokenCacheCredential - reads from the local MSAL cache (silent, no prompt)
        # 2. EnvironmentCredential, ManagedIdentityCredential, AzureCliCredential - non-interactive fallbacks
        # 3. InteractiveBrowserCredential - interactive login that writes to the cache for next time
        credential = ChainedTokenCredential(
            SharedTokenCacheCredential(cache_persistence_options=cache_options),
            EnvironmentCredential(),
            ManagedIdentityCredential(),
            AzureCliCredential(),
            InteractiveBrowserCredential(
                cache_persistence_options=cache_options,
            ),
        )

        kcsb = KustoConnectionStringBuilder.with_azure_token_credential(
            config.cluster_uri,
            credential,
        )

        client = ADXKustoClient(kcsb)
        self._client_cache[config.cluster_uri] = client
        self._client = client

    @property
    def database(self) -> str:
        return self._config.database

    @classmethod
    def from_env(cls) -> KustoClient:
        cluster_uri = cls._get_required_env("KUSTO_CLUSTER_URI")
        database = cls._get_required_env("KUSTO_DATABASE")

        config = KustoConfig(
            cluster_uri=cluster_uri,
            database=database,
        )
        return cls(config=config)

    @staticmethod
    def _get_required_env(name: str) -> str:
        import os

        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Environment variable {name} is required for Kusto configuration")
        return value

    def _default_request_properties(self) -> ClientRequestProperties:
        request_properties = ClientRequestProperties()
        request_properties.set_option("servertimeout", timedelta(minutes=2))
        request_properties.set_option("clientRequestId", f"langgraph-kusto-client;{str(uuid.uuid4())}")
        return request_properties

    def execute_query(self, query: str, *, properties: dict | None = None) -> Any:
        request_properties = self._default_request_properties()
        merged = dict(self._config.default_properties)
        if properties is not None:
            merged.update(properties)

        for key, value in merged.items():
            request_properties.set_option(key, value)

        return self._client.execute(self._config.database, query, request_properties)

    def execute_command(self, command: str, *, properties: dict | None = None) -> Any:
        request_properties = self._default_request_properties()
        merged = dict(self._config.default_properties)
        if properties is not None:
            merged.update(properties)

        for key, value in merged.items():
            request_properties.set_option(key, value)

        return self._client.execute(self._config.database, command, request_properties)

    async def execute_query_async(self, query: str, *, properties: dict | None = None) -> Any:
        raise NotImplementedError("Async Kusto query execution is not implemented yet.")

    async def execute_command_async(self, command: str, *, properties: dict | None = None) -> Any:
        raise NotImplementedError("Async Kusto command execution is not implemented yet.")

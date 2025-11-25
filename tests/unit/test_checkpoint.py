from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from langgraph_kusto.checkpoint.checkpoint import KustoCheckpointConfig, KustoCheckpointSaver


class TestKustoCheckpointSaver:
    """Unit tests for KustoCheckpointSaver with mocked KustoClient."""

    @pytest.fixture
    def mock_client(self):
        """Create a mocked KustoClient."""
        client = MagicMock()
        client.database = "TestDB"
        return client

    @pytest.fixture
    def saver(self, mock_client):
        """Create a KustoCheckpointSaver with mocked client."""
        config = KustoCheckpointConfig(
            client=mock_client,
            table_name="TestCheckpoints",
        )
        return KustoCheckpointSaver(config=config)

    def _mock_query_result(self, rows: list[dict]):
        """Helper to create a mock query result."""
        result = MagicMock()
        result.primary_results = [rows]
        return result

    def test_initialization(self, saver, mock_client):
        """Test that saver is initialized with correct table names."""
        assert saver._client == mock_client
        assert saver._table_name == "TestCheckpoints"
        assert saver._raw_table_name == "TestCheckpointsRaw"
        assert saver._writes_table_name == "TestCheckpointsWrites"
        assert saver._writes_raw_table_name == "TestCheckpointsWritesRaw"

    def test_put(self, saver, mock_client):
        """Test that put generates correct KQL command."""
        # Create a checkpoint
        checkpoint: Checkpoint = {
            "v": 1,
            "id": "checkpoint-1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"messages": []},
            "channel_versions": {},
            "versions_seen": {},
            "updated_channels": [],
        }
        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": 1,
            "parents": {},
        }
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }

        # Execute put
        result_config = saver.put(config, checkpoint, metadata, {})

        # Verify execute_command was called
        assert mock_client.execute_command.call_count == 1
        command_kql = mock_client.execute_command.call_args[0][0]

        # Verify KQL contains expected values
        assert ".set-or-append TestCheckpointsRaw <|" in command_kql
        assert 'ThreadId="thread-1"' in command_kql
        assert 'CheckpointNamespace=""' in command_kql
        assert 'CheckpointId="checkpoint-1"' in command_kql
        assert 'ParentCheckpointId=""' in command_kql
        assert "Deleted=false" in command_kql
        assert "Snapshot=dynamic(" in command_kql
        # Writes column should not be present in checkpoint table anymore
        assert "Writes=" not in command_kql.split("Snapshot=")[-1]

        # Verify returned config
        assert result_config["configurable"]["thread_id"] == "thread-1"
        assert result_config["configurable"]["checkpoint_id"] == "checkpoint-1"
        assert result_config["configurable"]["checkpoint_ns"] == ""

    def test_put_with_parent(self, saver, mock_client):
        """Test that put with parent checkpoint includes parent ID."""
        checkpoint: Checkpoint = {
            "v": 1,
            "id": "checkpoint-2",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"messages": []},
            "channel_versions": {},
            "versions_seen": {},
            "updated_channels": [],
        }
        metadata: CheckpointMetadata = {
            "source": "loop",
            "step": 2,
            "parents": {},
        }
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-1",  # Parent checkpoint
            }
        }

        # Execute put
        saver.put(config, checkpoint, metadata, {})

        # Verify command includes parent checkpoint ID
        command_kql = mock_client.execute_command.call_args[0][0]
        assert 'ParentCheckpointId="checkpoint-1"' in command_kql

    def test_put_writes(self, saver, mock_client):
        """Test that put_writes generates correct KQL command."""
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-1",
            }
        }
        writes = [("channel-1", "value-1"), ("channel-2", "value-2")]

        # Execute put_writes
        saver.put_writes(config, writes, "task-1")

        # Verify execute_command was called
        assert mock_client.execute_command.call_count == 1
        command_kql = mock_client.execute_command.call_args[0][0]

        # Verify KQL contains expected values for checkpoint writes table
        assert ".set-or-append TestCheckpointsWritesRaw <|" in command_kql
        assert 'ThreadId="thread-1"' in command_kql
        assert 'CheckpointId="checkpoint-1"' in command_kql
        assert 'TaskId="task-1"' in command_kql
        assert "Writes=" in command_kql
        # Snapshot should NOT be present in writes table
        assert "Snapshot=" not in command_kql

    def test_put_writes_no_checkpoint_id(self, saver, mock_client):
        """Test that put_writes returns early if no checkpoint_id."""
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }
        writes = [("channel-1", "value-1")]

        # Execute put_writes
        saver.put_writes(config, writes, "task-1")

        # Verify no command was executed
        assert mock_client.execute_command.call_count == 0

    def test_get_tuple_specific(self, saver, mock_client):
        """Test get_tuple with specific checkpoint ID."""
        # Mock query result for checkpoint
        mock_row = {
            "ThreadId": "thread-1",
            "CheckpointNamespace": "",
            "CheckpointId": "checkpoint-1",
            "ParentCheckpointId": "",
            "Snapshot": '{"v": 1, "id": "checkpoint-1", "ts": "2024-01-01T00:00:00Z", "channel_values": {}, "channel_versions": {}, "versions_seen": {}}',
            "CreatedAt": datetime.now(timezone.utc),
        }
        # Mock empty writes result
        mock_client.execute_query.side_effect = [
            self._mock_query_result([mock_row]),  # Checkpoint query
            self._mock_query_result([]),  # Writes query
        ]

        # Execute get_tuple
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-1",
            }
        }
        result = saver.get_tuple(config)

        # Verify two queries were executed (checkpoint + writes)
        assert mock_client.execute_query.call_count == 2

        # Verify checkpoint query
        checkpoint_query_kql = mock_client.execute_query.call_args_list[0][0][0]
        assert "TestCheckpoints()" in checkpoint_query_kql
        assert "ThreadId == 'thread-1'" in checkpoint_query_kql
        assert "CheckpointNamespace == ''" in checkpoint_query_kql
        assert "CheckpointId == 'checkpoint-1'" in checkpoint_query_kql
        assert "take 1" in checkpoint_query_kql

        # Verify writes query
        writes_query_kql = mock_client.execute_query.call_args_list[1][0][0]
        assert "TestCheckpointsWrites" in writes_query_kql
        assert "ThreadId == 'thread-1'" in writes_query_kql
        assert "CheckpointId == 'checkpoint-1'" in writes_query_kql

        # Verify result
        assert result is not None
        assert result.config["configurable"]["thread_id"] == "thread-1"
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint-1"
        assert result.checkpoint["id"] == "checkpoint-1"
        assert result.parent_config is None
        assert result.pending_writes is None

    def test_get_tuple_latest(self, saver, mock_client):
        """Test get_tuple without checkpoint ID retrieves latest."""
        # Mock query result for checkpoint
        mock_row = {
            "ThreadId": "thread-1",
            "CheckpointNamespace": "",
            "CheckpointId": "checkpoint-2",
            "ParentCheckpointId": "checkpoint-1",
            "Snapshot": '{"v": 1, "id": "checkpoint-2", "ts": "2024-01-01T00:00:00Z", "channel_values": {}, "channel_versions": {}, "versions_seen": {}}',
            "CreatedAt": datetime.now(timezone.utc),
        }
        # Mock empty writes result
        mock_client.execute_query.side_effect = [
            self._mock_query_result([mock_row]),  # Checkpoint query
            self._mock_query_result([]),  # Writes query
        ]

        # Execute get_tuple
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }
        result = saver.get_tuple(config)

        # Verify two queries were executed (checkpoint + writes)
        assert mock_client.execute_query.call_count == 2

        # Verify checkpoint query
        checkpoint_query_kql = mock_client.execute_query.call_args_list[0][0][0]
        assert "top 1 by CreatedAt desc" in checkpoint_query_kql

        # Verify result includes parent config
        assert result is not None
        assert result.config["configurable"]["checkpoint_id"] == "checkpoint-2"
        assert result.parent_config is not None
        assert result.parent_config["configurable"]["checkpoint_id"] == "checkpoint-1"

    def test_get_tuple_not_found(self, saver, mock_client):
        """Test get_tuple returns None when checkpoint not found."""
        # Mock empty result
        mock_client.execute_query.return_value = self._mock_query_result([])

        # Execute get_tuple
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "nonexistent",
            }
        }
        result = saver.get_tuple(config)

        # Verify result is None
        assert result is None

    def test_get_tuple_no_thread_id(self, saver, mock_client):
        """Test get_tuple returns None when no thread_id."""
        # Execute get_tuple
        config = {"configurable": {}}
        result = saver.get_tuple(config)

        # Verify no query was executed and result is None
        assert mock_client.execute_query.call_count == 0
        assert result is None

    def test_list(self, saver, mock_client):
        """Test list generates correct KQL and yields checkpoints."""
        # Mock query results
        mock_rows = [
            {
                "ThreadId": "thread-1",
                "CheckpointNamespace": "",
                "CheckpointId": "checkpoint-2",
                "ParentCheckpointId": "checkpoint-1",
                "Snapshot": '{"v": 1, "id": "checkpoint-2", "ts": "2024-01-01T00:00:00Z", "channel_values": {}, "channel_versions": {}, "versions_seen": {}}',
                "Writes": None,
                "CreatedAt": datetime.now(timezone.utc),
            },
            {
                "ThreadId": "thread-1",
                "CheckpointNamespace": "",
                "CheckpointId": "checkpoint-1",
                "ParentCheckpointId": "",
                "Snapshot": '{"v": 1, "id": "checkpoint-1", "ts": "2024-01-01T00:00:00Z", "channel_values": {}, "channel_versions": {}, "versions_seen": {}}',
                "Writes": None,
                "CreatedAt": datetime.now(timezone.utc),
            },
        ]
        mock_client.execute_query.return_value = self._mock_query_result(mock_rows)

        # Execute list
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }
        results = list(saver.list(config, limit=10))

        # Verify query was executed
        assert mock_client.execute_query.call_count == 1
        query_kql = mock_client.execute_query.call_args[0][0]

        # Verify KQL contains expected filters
        assert "TestCheckpoints()" in query_kql
        assert "ThreadId == 'thread-1'" in query_kql
        assert "order by CreatedAt desc" in query_kql
        assert "take 10" in query_kql

        # Verify results
        assert len(results) == 2
        assert results[0].config["configurable"]["checkpoint_id"] == "checkpoint-2"
        assert results[1].config["configurable"]["checkpoint_id"] == "checkpoint-1"

    def test_list_with_before(self, saver, mock_client):
        """Test list with before filter."""
        # Mock before checkpoint query
        before_time = datetime.now(timezone.utc)
        mock_client.execute_query.side_effect = [
            self._mock_query_result([{"CreatedAt": before_time}]),  # Before checkpoint
            self._mock_query_result([]),  # Main list query
        ]

        # Execute list
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
            }
        }
        before_config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-2",
            }
        }
        list(saver.list(config, before=before_config))

        # Verify two queries were executed
        assert mock_client.execute_query.call_count == 2

        # Verify second query includes time filter
        query_kql = mock_client.execute_query.call_args_list[1][0][0]
        assert f"CreatedAt < datetime({before_time.isoformat()})" in query_kql

    def test_get_tuple_with_pending_writes(self, saver, mock_client):
        """Test get_tuple retrieves both checkpoint and pending writes."""
        # Mock query result for checkpoint
        mock_checkpoint_row = {
            "ThreadId": "thread-1",
            "CheckpointNamespace": "",
            "CheckpointId": "checkpoint-1",
            "ParentCheckpointId": "",
            "Snapshot": '{"v": 1, "id": "checkpoint-1", "ts": "2024-01-01T00:00:00Z", "channel_values": {}, "channel_versions": {}, "versions_seen": {}}',
            "CreatedAt": datetime.now(timezone.utc),
        }
        # Mock writes result
        mock_writes_row = {
            "ThreadId": "thread-1",
            "CheckpointNamespace": "",
            "CheckpointId": "checkpoint-1",
            "TaskId": "task-1",
            "Writes": '[["channel-1", "value-1"], ["channel-2", "value-2"]]',
            "CreatedAt": datetime.now(timezone.utc),
        }
        mock_client.execute_query.side_effect = [
            self._mock_query_result([mock_checkpoint_row]),  # Checkpoint query
            self._mock_query_result([mock_writes_row]),  # Writes query
        ]

        # Execute get_tuple
        config = {
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "checkpoint-1",
            }
        }
        result = saver.get_tuple(config)

        # Verify result includes pending writes
        assert result is not None
        assert result.checkpoint["id"] == "checkpoint-1"
        assert result.pending_writes is not None
        assert len(result.pending_writes) == 2
        assert result.pending_writes[0] == ("channel-1", "value-1")
        assert result.pending_writes[1] == ("channel-2", "value-2")

    def test_delete_thread(self, saver, mock_client):
        """Test delete_thread inserts deletion marker."""
        # Execute delete
        saver.delete_thread("thread-1")

        # Verify execute_command was called
        assert mock_client.execute_command.call_count == 1
        command_kql = mock_client.execute_command.call_args[0][0]

        # Verify KQL contains deletion marker
        assert ".set-or-append TestCheckpointsRaw <|" in command_kql
        assert 'ThreadId="thread-1"' in command_kql
        assert "Deleted=true" in command_kql
        assert 'CheckpointNamespace=""' in command_kql
        assert 'CheckpointId=""' in command_kql
        assert "Snapshot=dynamic({})" in command_kql
        # Writes column should not be present
        assert "Writes=" not in command_kql.split("Snapshot=")[-1]

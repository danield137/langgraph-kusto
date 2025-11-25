from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from typing_extensions import Protocol


@dataclass(slots=True)
class KustoConfig:
    cluster_uri: str
    database: str
    default_properties: dict = field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

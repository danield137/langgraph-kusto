from typing import Protocol, Sequence
from langchain_core.messages import BaseMessage


class MemoryStrategy(Protocol):
    """Protocol defining the interface for memory strategies.
    
    Memory strategies handle both retrieval (recall) and persistence (remember)
    of conversational context.
    """
    
    def recall(
        self,
        *,
        store,
        user_id: str,
        messages: Sequence[BaseMessage]
    ) -> list[dict]:
        """Execute memory retrieval.
        
        Decide what query to run and execute it against the store.
        
        Args:
            store: BaseStore instance for memory persistence
            user_id: Unique identifier for the user
            messages: Sequence of messages in current conversation
            
        Returns:
            List of raw memory item dicts (e.g. {"content": str, "type": str, ...})
        """
        ...
    
    def remember(
        self,
        *,
        store,
        user_id: str,
        last_user_msg: BaseMessage | None,
        messages: Sequence[BaseMessage]
    ) -> None:
        """Decide what to persist and write to store.
        
        Args:
            store: BaseStore instance for memory persistence
            user_id: Unique identifier for the user
            last_user_msg: The most recent user message
            messages: Sequence of messages in current conversation
        """
        ...

import uuid
from langchain_core.messages import HumanMessage


class KeywordMemoryStrategy:
    """Memory strategy that stores and recalls based on keyword matching.
    
    This strategy:
    - Stores user preferences when preference keywords are detected
    - Stores short follow-up details (elaborations on previous statements)
    - Recalls memories using semantic search (if store supports it) or basic retrieval
    """
    
    def __init__(
        self,
        preference_keywords: list[str] | None = None,
        short_detail_max_words: int = 5,
        base_namespace: tuple[str, ...] = ("memories",),
        limit: int = 50
    ):
        """Initialize keyword-based memory strategy.
        
        Args:
            preference_keywords: Keywords that indicate user preferences
            short_detail_max_words: Maximum word count for short detail storage
            base_namespace: Base namespace tuple for memory storage
            limit: Maximum number of memories to recall
        """
        self.preference_keywords = preference_keywords or [
            "like", "love", "favorite", "prefer", "enjoy"
        ]
        self.short_detail_max_words = short_detail_max_words
        self.base_namespace = base_namespace
        self.limit = limit
    
    def _namespace(self, user_id: str) -> tuple[str, ...]:
        """Build namespace for user."""
        return self.base_namespace + (user_id,)
    
    def recall(self, *, store, user_id: str, messages):
        """Retrieve memories from store.
        
        Uses semantic search if last message is present and store supports it.
        """
        # Derive semantic query from last user message if available
        query = None
        if messages and isinstance(getattr(messages[-1], "content", None), str):
            text = messages[-1].content.strip()
            if len(text) > 3:
                query = text  # Store will ignore if no embedding index configured
        
        results = store.search(
            self._namespace(user_id),
            query=query,
            limit=self.limit
        )
        
        # Return raw memory dicts
        return [r.value for r in results]
    
    def remember(self, *, store, user_id: str, last_user_msg, messages):
        """Store important information from user message.
        
        Stores preferences and short follow-up details.
        """
        if not isinstance(last_user_msg, HumanMessage):
            return
        
        content = last_user_msg.content
        if not isinstance(content, str):
            return
        
        lower = content.lower()
        ns = self._namespace(user_id)
        
        # Store user preferences (like, love, favorite, prefer, enjoy)
        if any(keyword in lower for keyword in self.preference_keywords):
            memory_key = f"preference_{uuid.uuid4()}"
            store.put(ns, memory_key, {"content": content, "type": "preference"})
        
        # Store short follow-up details (likely elaborating on previous statement)
        elif len(lower.split()) <= self.short_detail_max_words:
            memory_key = f"detail_{uuid.uuid4()}"
            store.put(
                ns,
                memory_key,
                {"content": f"Additional detail: {content}", "type": "detail"}
            )

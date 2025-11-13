class NoMemoryStrategy:
    """Memory strategy that doesn't store or recall any information.
    
    This is a stateless strategy useful for chatbots that don't need
    to remember previous conversations.
    """
    
    def recall(self, *, store, user_id: str, messages):
        """Return empty list - no memory to recall.
        
        Args:
            store: BaseStore (unused in this strategy)
            user_id: User identifier (unused in this strategy)
            messages: Message sequence (unused in this strategy)
            
        Returns:
            Empty list
        """
        return []
    
    def remember(self, *, store, user_id: str, last_user_msg, messages):
        """No-op - nothing to store.
        
        Args:
            store: BaseStore (unused in this strategy)
            user_id: User identifier (unused in this strategy)
            last_user_msg: Last user message (unused in this strategy)
            messages: Message sequence (unused in this strategy)
        """
        pass

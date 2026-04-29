class ChatPLVisError(Exception):
    """Base exception for all ChatPLVis errors."""


class DataError(ChatPLVisError):
    """Raised when CSV data cannot be loaded or is malformed."""


class EmbeddingError(ChatPLVisError):
    """Raised when the embedding model or FAISS index fails."""


class LLMError(ChatPLVisError):
    """Raised when the LLM service returns an error."""


class RateLimitedError(LLMError):
    """Raised when the LLM service rate-limits the request."""


class ValidationError(ChatPLVisError):
    """Raised when request input fails validation."""

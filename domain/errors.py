class DomainError(Exception):
    """Base class for all explicitly modelled domain errors."""


class EmptyDocumentError(DomainError):
    """Raised when a document has no extractable text."""


class UnsupportedFormatError(DomainError):
    """Raised when the ingestion adapter cannot parse the given doc_type."""


class ChunkingError(DomainError):
    """Raised when a document cannot be chunked (e.g. degenerate structure)."""

# Base Parser - Abstract base class for all file parsers

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ContentSection:
    """
    Represents a content-bearing section extracted from a document.
    
    IMPORTANT: Only sections with actual text content should be created.
    Empty headers or structural elements without content are NOT valid sections.
    """
    text: str                              # The actual content (MUST be non-empty)
    title: Optional[str] = None            # Section title/header (if any)
    hierarchy: List[str] = field(default_factory=list)  # Path: ["H1", "H2", "H3"]
    start_char: int = 0                    # Position in original document
    end_char: int = 0                      # End position in original document
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra info
    
    def __post_init__(self):
        """Validate that section has actual content."""
        if not self.text or not self.text.strip():
            raise ValueError("ContentSection must have non-empty text content")
    
    @property
    def full_title(self) -> str:
        """Get full hierarchical title."""
        if self.hierarchy:
            return " > ".join(self.hierarchy)
        return self.title or ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "title": self.title,
            "hierarchy": self.hierarchy,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


class BaseParser(ABC):
    """
    Abstract base class for file parsers.
    
    All parsers must implement the `parse` method which extracts
    content-bearing sections from raw file content.
    
    IMPORTANT RULES:
    1. Only create ContentSection for text that has actual content
    2. Empty headers/titles without content should NOT create sections
    3. Preserve hierarchy information in metadata, not as separate nodes
    """
    
    @abstractmethod
    def parse(self, content: str, filename: str = "") -> List[ContentSection]:
        """
        Parse file content and extract content-bearing sections.
        
        Args:
            content: Raw file content as string
            filename: Original filename (for metadata)
            
        Returns:
            List of ContentSection objects, each with non-empty text
        """
        pass
    
    @abstractmethod
    def get_file_type(self) -> str:
        """Return the file type this parser handles."""
        pass
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        # Normalize whitespace
        text = " ".join(text.split())
        return text.strip()
    
    def _is_valid_content(self, text: str) -> bool:
        """Check if text has meaningful content."""
        if not text:
            return False
        cleaned = self._clean_text(text)
        # Must have at least some words (not just punctuation)
        return len(cleaned) > 10 and any(c.isalnum() for c in cleaned)

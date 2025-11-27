# Text Parser - For plain text files

import re
from typing import List
from .base import BaseParser, ContentSection

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class TextParser(BaseParser):
    """
    Parser for plain text files.
    
    Uses NLTK for sentence boundary detection if available.
    Falls back to simple paragraph/sentence splitting.
    
    Key behavior:
    - Treats double newlines as paragraph breaks
    - Each paragraph becomes a section
    - Preserves original text without modification
    """
    
    def __init__(self):
        if NLTK_AVAILABLE:
            # Download punkt tokenizer if needed
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                except:
                    pass  # Will fall back to regex
    
    def get_file_type(self) -> str:
        return "text"
    
    def parse(self, content: str, filename: str = "") -> List[ContentSection]:
        """Parse plain text content into sections."""
        if not content or not content.strip():
            return []
        
        sections = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', content)
        
        char_pos = 0
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if self._is_valid_content(para):
                sections.append(ContentSection(
                    text=para,
                    title=None,
                    hierarchy=[],
                    start_char=char_pos,
                    end_char=char_pos + len(para),
                    metadata={
                        "source": filename, 
                        "format": "text",
                        "paragraph_index": i
                    }
                ))
            char_pos += len(para) + 2  # +2 for the double newline
        
        # If no paragraphs found (single block of text), treat as one section
        if not sections and self._is_valid_content(content):
            sections.append(ContentSection(
                text=content.strip(),
                title=None,
                hierarchy=[],
                start_char=0,
                end_char=len(content),
                metadata={"source": filename, "format": "text"}
            ))
        
        return sections
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences. Used by chunker."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback: simple regex-based sentence splitting
        # Split on .!? followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

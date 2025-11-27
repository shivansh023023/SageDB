# Semantic Chunker - Splits content into embedding-friendly chunks

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, MIN_CHUNK_SIZE


@dataclass
class Chunk:
    """Represents a text chunk ready for embedding."""
    text: str
    chunk_index: int
    token_count: int
    start_char: int
    end_char: int
    title: Optional[str] = None
    hierarchy: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "title": self.title,
            "hierarchy": self.hierarchy,
            "metadata": self.metadata,
        }


class SemanticChunker:
    """
    Splits text into semantic chunks suitable for embedding.
    
    Key features:
    - Respects sentence boundaries
    - Configurable chunk size and overlap
    - Preserves metadata through chunking
    """
    
    def __init__(self, 
                 max_tokens: int = DEFAULT_CHUNK_SIZE,
                 overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
                 min_tokens: int = MIN_CHUNK_SIZE):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        
        # Initialize NLTK if available
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                    nltk.download('punkt_tab', quiet=True)
                except:
                    pass
    
    def chunk_text(self, text: str, title: Optional[str] = None,
                   hierarchy: Optional[List[str]] = None,
                   metadata: Optional[dict] = None) -> List[Chunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            title: Optional title to include in each chunk's metadata
            hierarchy: Optional hierarchy path
            metadata: Optional metadata to preserve
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        token_count = self._estimate_tokens(text)
        
        # If text fits in one chunk, return as-is
        if token_count <= self.max_tokens:
            return [Chunk(
                text=text,
                chunk_index=0,
                token_count=token_count,
                start_char=0,
                end_char=len(text),
                title=title,
                hierarchy=hierarchy or [],
                metadata=metadata or {}
            )]
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        # Build chunks from sentences
        chunks = []
        current_sentences: List[str] = []
        current_tokens = 0
        chunk_start = 0
        char_pos = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If single sentence exceeds max, we need to split it
            if sentence_tokens > self.max_tokens:
                # First, save current chunk if any
                if current_sentences:
                    chunk_text = ' '.join(current_sentences)
                    chunks.append(Chunk(
                        text=chunk_text,
                        chunk_index=len(chunks),
                        token_count=current_tokens,
                        start_char=chunk_start,
                        end_char=char_pos,
                        title=title,
                        hierarchy=hierarchy or [],
                        metadata=metadata or {}
                    ))
                    current_sentences = []
                    current_tokens = 0
                
                # Split the long sentence by words
                words = sentence.split()
                word_chunk = []
                word_tokens = 0
                
                for word in words:
                    word_token_count = self._estimate_tokens(word)
                    if word_tokens + word_token_count > self.max_tokens and word_chunk:
                        chunk_text = ' '.join(word_chunk)
                        chunks.append(Chunk(
                            text=chunk_text,
                            chunk_index=len(chunks),
                            token_count=word_tokens,
                            start_char=char_pos,
                            end_char=char_pos + len(chunk_text),
                            title=title,
                            hierarchy=hierarchy or [],
                            metadata=metadata or {}
                        ))
                        
                        # Overlap: keep some words
                        overlap_words = max(1, int(len(word_chunk) * (self.overlap_tokens / self.max_tokens)))
                        word_chunk = word_chunk[-overlap_words:]
                        word_tokens = self._estimate_tokens(' '.join(word_chunk))
                    
                    word_chunk.append(word)
                    word_tokens += word_token_count
                
                # Add remaining words to next chunk consideration
                if word_chunk:
                    current_sentences = [' '.join(word_chunk)]
                    current_tokens = word_tokens
                    chunk_start = char_pos
            
            # Check if adding this sentence would exceed max
            elif current_tokens + sentence_tokens > self.max_tokens:
                # Save current chunk
                if current_sentences:
                    chunk_text = ' '.join(current_sentences)
                    chunks.append(Chunk(
                        text=chunk_text,
                        chunk_index=len(chunks),
                        token_count=current_tokens,
                        start_char=chunk_start,
                        end_char=char_pos,
                        title=title,
                        hierarchy=hierarchy or [],
                        metadata=metadata or {}
                    ))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_sentences, self.overlap_tokens
                )
                current_sentences = overlap_sentences + [sentence]
                current_tokens = self._estimate_tokens(' '.join(current_sentences))
                chunk_start = char_pos
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
            
            char_pos += len(sentence) + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunk_tokens = self._estimate_tokens(chunk_text)
            
            # Only add if it meets minimum size OR it's the only chunk
            if chunk_tokens >= self.min_tokens or not chunks:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    token_count=chunk_tokens,
                    start_char=chunk_start,
                    end_char=len(text),
                    title=title,
                    hierarchy=hierarchy or [],
                    metadata=metadata or {}
                ))
            elif chunks:
                # Append to previous chunk if too small
                last_chunk = chunks[-1]
                combined_text = last_chunk.text + ' ' + chunk_text
                chunks[-1] = Chunk(
                    text=combined_text,
                    chunk_index=last_chunk.chunk_index,
                    token_count=self._estimate_tokens(combined_text),
                    start_char=last_chunk.start_char,
                    end_char=len(text),
                    title=title,
                    hierarchy=hierarchy or [],
                    metadata=metadata or {}
                )
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except Exception:
                pass
        
        # Fallback: regex-based sentence splitting
        # Split on .!? followed by space and capital letter or end of string
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Rough estimate: ~4 characters per token for English.
        This is a common heuristic for transformer models.
        """
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def _get_overlap_sentences(self, sentences: List[str], 
                               target_tokens: int) -> List[str]:
        """Get sentences for overlap from the end of current chunk."""
        if not sentences or target_tokens <= 0:
            return []
        
        overlap = []
        tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if tokens + sentence_tokens <= target_tokens:
                overlap.insert(0, sentence)
                tokens += sentence_tokens
            else:
                break
        
        return overlap

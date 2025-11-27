# Text Parser - Implements "Sticky Header" pattern with heuristics
# Detects headers via patterns: colons, short lines, ALL CAPS, etc.

import re
from typing import List, Optional, Tuple
from .base import BaseParser, ContentSection


class TextParser(BaseParser):
    """
    Parser for plain text files with Sticky Header pattern.
    
    Header detection heuristics:
    1. Lines ending with colon "Priority:" 
    2. Short lines (< 60 chars) followed by longer content
    3. ALL CAPS lines "INSTALLATION GUIDE"
    4. Lines starting with numbers "1. First Step" (as headers for sections)
    5. Lines with common header patterns "Chapter X:", "Section X"
    
    Key behavior:
    - Detected header acts as MAGNET
    - Following content GLUED until next header
    - Header text INCLUDED in section content
    
    Example:
        Input:
            "Priority for hackathon:
            
            We need to focus on:
            - Ingestion pipeline
            - Storage layer
            - Retrieval"
        
        Output:
            Section: "Priority for hackathon:\n\nWe need to focus on:\n- Ingestion pipeline\n- Storage layer\n- Retrieval"
    """
    
    # Patterns that indicate a line might be a header
    HEADER_PATTERNS = [
        # Line ending with colon (common header pattern)
        re.compile(r'^.{3,60}:\s*$'),
        # ALL CAPS words (at least 3 words, 10+ chars)
        re.compile(r'^[A-Z][A-Z\s]{10,}$'),
        # Numbered sections like "1. Introduction" or "1) First"
        re.compile(r'^\d+[\.\)]\s+[A-Z]'),
        # Chapter/Section patterns
        re.compile(r'^(Chapter|Section|Part|Module|Unit|Appendix)\s+(\d+|[IVXLC]+)', re.IGNORECASE),
        # Markdown-style headers in plain text
        re.compile(r'^#{1,6}\s+.+'),
        # Underlined headers (line of === or ---)
        re.compile(r'^[=\-]{3,}$'),
    ]
    
    # Max chars for a line to be considered a potential header
    MAX_HEADER_LENGTH = 80
    
    def get_file_type(self) -> str:
        return "text"
    
    def parse(self, content: str, filename: str = "") -> List[ContentSection]:
        """Parse plain text using sticky header pattern with heuristics."""
        if not content or not content.strip():
            return []
        
        lines = content.split('\n')
        sections = []
        
        # First pass: identify likely headers
        header_lines = self._identify_headers(lines)
        
        # Second pass: build sections with sticky headers
        current_title: Optional[str] = None
        current_hierarchy: List[str] = []
        buffer_parts: List[str] = []
        section_start = 0
        
        def flush_section():
            nonlocal buffer_parts, section_start
            if buffer_parts:
                section_text = '\n'.join(buffer_parts).strip()
                if self._is_valid_content(section_text):
                    sections.append(ContentSection(
                        text=section_text,
                        title=current_title,
                        hierarchy=current_hierarchy.copy(),
                        start_char=section_start,
                        end_char=section_start + len(section_text),
                        metadata={"source": filename, "format": "text"}
                    ))
                buffer_parts = []
        
        char_pos = 0
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if i in header_lines:
                # This is a header - flush previous section
                flush_section()
                section_start = char_pos
                
                # Update hierarchy (simple: just use the header text)
                if line_stripped:
                    # Remove trailing colon for hierarchy
                    clean_title = line_stripped.rstrip(':').strip()
                    current_title = clean_title
                    # Simple hierarchy: just current header
                    current_hierarchy = [clean_title]
                
                # STICKY: Include header in buffer
                buffer_parts = [line_stripped] if line_stripped else []
            else:
                # Regular content line - glue to current section
                if line_stripped:
                    buffer_parts.append(line)
                elif buffer_parts:
                    # Preserve empty lines within content
                    buffer_parts.append('')
            
            char_pos += len(line) + 1  # +1 for newline
        
        # Flush remaining content
        flush_section()
        
        # If no headers detected, treat entire content as one section
        if not sections and self._is_valid_content(content.strip()):
            sections.append(ContentSection(
                text=content.strip(),
                title=None,
                hierarchy=[],
                start_char=0,
                end_char=len(content),
                metadata={"source": filename, "format": "text"}
            ))
        
        return sections
    
    def _identify_headers(self, lines: List[str]) -> set:
        """Identify line indices that are likely headers."""
        headers = set()
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Skip very long lines
            if len(line_stripped) > self.MAX_HEADER_LENGTH:
                continue
            
            # Check against header patterns
            for pattern in self.HEADER_PATTERNS:
                if pattern.match(line_stripped):
                    headers.add(i)
                    break
            
            # Check for underlined headers (next line is === or ---)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^[=\-]{3,}$', next_line) and len(next_line) >= len(line_stripped) * 0.5:
                    headers.add(i)
                    headers.add(i + 1)  # Mark underline as part of header
            
            # Heuristic: Short line followed by blank then content
            if (len(line_stripped) < 50 and 
                not line_stripped.endswith(('.', ',', ';')) and
                i + 2 < len(lines) and
                not lines[i + 1].strip() and
                lines[i + 2].strip()):
                # Check if first char of next content is lowercase (indicates continuation)
                next_content = lines[i + 2].strip()
                if next_content and next_content[0].isupper():
                    headers.add(i)
        
        return headers

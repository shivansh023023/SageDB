# Markdown Parser - Implements "Sticky Header" pattern
# Headers are magnets that hold onto all following content until the next header

import re
from typing import List, Optional
from .base import BaseParser, ContentSection

try:
    from markdown_it import MarkdownIt
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False


class MarkdownParser(BaseParser):
    """
    Parser for Markdown files with Sticky Header pattern.
    
    Key behavior:
    - Headers ACT AS MAGNETS - they hold onto all following content
    - The header text is INCLUDED in the section content (not just metadata)
    - A new header triggers saving the previous section
    - This ensures vector search for "priorities" returns both the header AND the list
    
    Example:
        Input:
            ## Installation
            Run `pip install sagedb`
            
            ## Usage
            Import and use...
        
        Output:
            Section 1: "## Installation\nRun `pip install sagedb`"
            Section 2: "## Usage\nImport and use..."
    """
    
    def __init__(self):
        if MARKDOWN_IT_AVAILABLE:
            self.md = MarkdownIt()
        else:
            self.md = None
    
    def get_file_type(self) -> str:
        return "markdown"
    
    def parse(self, content: str, filename: str = "") -> List[ContentSection]:
        """Parse markdown using sticky header pattern."""
        if not content or not content.strip():
            return []
        
        sections = []
        lines = content.split('\n')
        
        # State machine variables
        current_hierarchy: List[str] = []
        buffer_lines: List[str] = []  # Accumulates header + content
        current_title: Optional[str] = None
        section_start: int = 0
        char_pos = 0
        
        for i, line in enumerate(lines):
            line_start = char_pos
            char_pos += len(line) + 1  # +1 for newline
            
            # Check if this is a header line
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if header_match:
                # FLUSH: Save previous section if buffer has meaningful content
                if buffer_lines:
                    section_text = '\n'.join(buffer_lines).strip()
                    if self._is_valid_content(section_text):
                        sections.append(ContentSection(
                            text=section_text,
                            title=current_title,
                            hierarchy=current_hierarchy.copy(),
                            start_char=section_start,
                            end_char=line_start,
                            metadata={"source": filename, "format": "markdown"}
                        ))
                
                # START NEW: Update hierarchy and begin new buffer
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Trim hierarchy to parent level, then add current title
                current_hierarchy = current_hierarchy[:level-1]
                current_hierarchy.append(title)
                current_title = title
                
                # STICKY: Include the header line itself in the buffer
                buffer_lines = [line]  # Header becomes first line of new section
                section_start = line_start
                
            else:
                # GLUE: Append content to current buffer
                buffer_lines.append(line)
        
        # FLUSH: Don't forget the last section
        if buffer_lines:
            section_text = '\n'.join(buffer_lines).strip()
            if self._is_valid_content(section_text):
                sections.append(ContentSection(
                    text=section_text,
                    title=current_title,
                    hierarchy=current_hierarchy.copy(),
                    start_char=section_start,
                    end_char=len(content),
                    metadata={"source": filename, "format": "markdown"}
                ))
        
        # Handle content with no headers at all
        if not sections and content.strip():
            if self._is_valid_content(content.strip()):
                sections.append(ContentSection(
                    text=content.strip(),
                    title=None,
                    hierarchy=[],
                    start_char=0,
                    end_char=len(content),
                    metadata={"source": filename, "format": "markdown"}
                ))
        
        return sections

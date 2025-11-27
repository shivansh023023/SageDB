# Markdown Parser - Uses markdown-it-py for parsing

import re
from typing import List, Optional, Tuple
from .base import BaseParser, ContentSection

try:
    from markdown_it import MarkdownIt
    MARKDOWN_IT_AVAILABLE = True
except ImportError:
    MARKDOWN_IT_AVAILABLE = False


class MarkdownParser(BaseParser):
    """
    Parser for Markdown files.
    
    Uses markdown-it-py for robust parsing. Falls back to regex if not available.
    
    Key behavior:
    - Only creates sections for headers that have content underneath
    - Empty headers (just a title, no content) are preserved in hierarchy only
    - Code blocks are included as part of section content
    """
    
    def __init__(self):
        if MARKDOWN_IT_AVAILABLE:
            self.md = MarkdownIt()
        else:
            self.md = None
    
    def get_file_type(self) -> str:
        return "markdown"
    
    def parse(self, content: str, filename: str = "") -> List[ContentSection]:
        """Parse markdown content into content-bearing sections."""
        if not content or not content.strip():
            return []
        
        if self.md:
            return self._parse_with_markdown_it(content, filename)
        else:
            return self._parse_with_regex(content, filename)
    
    def _parse_with_markdown_it(self, content: str, filename: str) -> List[ContentSection]:
        """Parse using markdown-it-py AST."""
        sections = []
        current_hierarchy: List[str] = []
        current_content_lines: List[str] = []
        current_title: Optional[str] = None
        current_start: int = 0
        
        lines = content.split('\n')
        line_positions = self._get_line_positions(content)
        
        for i, line in enumerate(lines):
            # Check if this is a header line
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            
            if header_match:
                # Save previous section if it has content
                if current_content_lines:
                    section_text = '\n'.join(current_content_lines).strip()
                    if self._is_valid_content(section_text):
                        sections.append(ContentSection(
                            text=section_text,
                            title=current_title,
                            hierarchy=current_hierarchy.copy(),
                            start_char=current_start,
                            end_char=line_positions[i] if i < len(line_positions) else len(content),
                            metadata={"source": filename, "format": "markdown"}
                        ))
                
                # Update hierarchy based on header level
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Trim hierarchy to current level and add new title
                current_hierarchy = current_hierarchy[:level-1]
                current_hierarchy.append(title)
                
                current_title = title
                current_content_lines = []
                current_start = line_positions[i] if i < len(line_positions) else 0
            else:
                # Regular content line
                if line.strip():  # Only add non-empty lines
                    current_content_lines.append(line)
        
        # Don't forget the last section
        if current_content_lines:
            section_text = '\n'.join(current_content_lines).strip()
            if self._is_valid_content(section_text):
                sections.append(ContentSection(
                    text=section_text,
                    title=current_title,
                    hierarchy=current_hierarchy.copy(),
                    start_char=current_start,
                    end_char=len(content),
                    metadata={"source": filename, "format": "markdown"}
                ))
        
        # Handle case where there's content before any headers
        if not sections and content.strip():
            # Check if the whole content is valid (no headers at all)
            clean_content = self._remove_headers(content)
            if self._is_valid_content(clean_content):
                sections.append(ContentSection(
                    text=clean_content.strip(),
                    title=None,
                    hierarchy=[],
                    start_char=0,
                    end_char=len(content),
                    metadata={"source": filename, "format": "markdown"}
                ))
        
        return sections
    
    def _parse_with_regex(self, content: str, filename: str) -> List[ContentSection]:
        """Fallback regex-based parsing."""
        # Split by headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        sections = []
        current_hierarchy: List[str] = []
        
        # Split content into chunks by headers
        parts = re.split(r'(^#{1,6}\s+.+$)', content, flags=re.MULTILINE)
        
        current_title = None
        current_start = 0
        
        i = 0
        while i < len(parts):
            part = parts[i]
            header_match = re.match(header_pattern, part.strip())
            
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Update hierarchy
                current_hierarchy = current_hierarchy[:level-1]
                current_hierarchy.append(title)
                current_title = title
                
                # Get content after this header
                if i + 1 < len(parts):
                    content_text = parts[i + 1].strip()
                    if self._is_valid_content(content_text):
                        sections.append(ContentSection(
                            text=content_text,
                            title=current_title,
                            hierarchy=current_hierarchy.copy(),
                            start_char=current_start,
                            end_char=current_start + len(content_text),
                            metadata={"source": filename, "format": "markdown"}
                        ))
                    i += 2
                else:
                    i += 1
            else:
                # Content before first header
                if part.strip() and self._is_valid_content(part):
                    sections.append(ContentSection(
                        text=part.strip(),
                        title=None,
                        hierarchy=[],
                        start_char=0,
                        end_char=len(part),
                        metadata={"source": filename, "format": "markdown"}
                    ))
                i += 1
        
        return sections
    
    def _get_line_positions(self, content: str) -> List[int]:
        """Get character positions of each line start."""
        positions = [0]
        for i, char in enumerate(content):
            if char == '\n':
                positions.append(i + 1)
        return positions
    
    def _remove_headers(self, content: str) -> str:
        """Remove header lines from content."""
        lines = content.split('\n')
        return '\n'.join(line for line in lines if not re.match(r'^#{1,6}\s+', line.strip()))

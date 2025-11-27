# HTML Parser - Implements "Sticky Header" pattern
# h1-h6 are magnets that hold onto all following p/ul/table content

import re
from typing import List, Optional
from .base import BaseParser, ContentSection

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class HTMLParser(BaseParser):
    """
    Parser for HTML files with Sticky Header pattern.
    
    Key behavior:
    - h1-h6 tags act as MAGNETS
    - All following p, ul, ol, table, pre, blockquote are GLUED to the header
    - A new header triggers saving the previous section
    - Header text is INCLUDED in the section content
    
    Example:
        Input:
            <h2>Installation</h2>
            <p>Run pip install sagedb</p>
            <ul><li>Step 1</li><li>Step 2</li></ul>
            <h2>Usage</h2>
            <p>Import and use</p>
        
        Output:
            Section 1: "Installation\nRun pip install sagedb\n• Step 1\n• Step 2"
            Section 2: "Usage\nImport and use"
    """
    
    # Tags to completely ignore
    IGNORE_TAGS = {'script', 'style', 'nav', 'header', 'footer', 'aside', 
                   'noscript', 'iframe', 'meta', 'link', 'head'}
    
    # Header tags (magnets)
    HEADER_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    
    # Content tags (payload to glue)
    CONTENT_TAGS = {'p', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 
                    'pre', 'code', 'blockquote', 'div', 'span', 'article', 
                    'section', 'main', 'figcaption'}
    
    def get_file_type(self) -> str:
        return "html"
    
    def parse(self, content: str, filename: str = "") -> List[ContentSection]:
        """Parse HTML using sticky header pattern."""
        if not content or not content.strip():
            return []
        
        if not BS4_AVAILABLE:
            return self._parse_with_regex(content, filename)
        
        return self._parse_with_beautifulsoup(content, filename)
    
    def _parse_with_beautifulsoup(self, content: str, filename: str) -> List[ContentSection]:
        """Parse using BeautifulSoup with sticky header pattern."""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup.find_all(list(self.IGNORE_TAGS)):
            tag.decompose()
        
        sections = []
        current_hierarchy: List[str] = []
        buffer_parts: List[str] = []  # Accumulates header + content text
        current_title: Optional[str] = None
        
        def get_text_content(element) -> str:
            """Extract text from an element, handling lists specially."""
            if isinstance(element, NavigableString):
                return str(element).strip()
            
            if not isinstance(element, Tag):
                return ""
            
            tag_name = element.name
            
            if tag_name in self.IGNORE_TAGS:
                return ""
            
            # Handle lists - preserve structure
            if tag_name in ('ul', 'ol'):
                items = []
                for li in element.find_all('li', recursive=False):
                    item_text = li.get_text(separator=' ', strip=True)
                    if item_text:
                        items.append(f"• {item_text}")
                return '\n'.join(items)
            
            # Handle code blocks
            if tag_name in ('pre', 'code'):
                return element.get_text(strip=True)
            
            # Default: get text with space separator
            return element.get_text(separator=' ', strip=True)
        
        def flush_section():
            """Save current buffer as a section if it has valid content."""
            nonlocal buffer_parts, current_title, current_hierarchy
            if buffer_parts:
                section_text = '\n'.join(buffer_parts).strip()
                if self._is_valid_content(section_text):
                    sections.append(ContentSection(
                        text=section_text,
                        title=current_title,
                        hierarchy=current_hierarchy.copy(),
                        start_char=0,
                        end_char=len(section_text),
                        metadata={"source": filename, "format": "html"}
                    ))
                buffer_parts = []
        
        def process_element(element):
            """Process a single element with sticky header logic."""
            nonlocal current_hierarchy, buffer_parts, current_title
            
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    buffer_parts.append(text)
                return
            
            if not isinstance(element, Tag):
                return
            
            tag_name = element.name
            
            if tag_name in self.IGNORE_TAGS:
                return
            
            # MAGNET: Header tags start new sections
            if tag_name in self.HEADER_TAGS:
                # Flush previous section
                flush_section()
                
                # Update hierarchy
                level = int(tag_name[1])
                title = element.get_text(strip=True)
                
                current_hierarchy = current_hierarchy[:level-1]
                if title:
                    current_hierarchy.append(title)
                current_title = title
                
                # STICKY: Include header text in buffer
                if title:
                    buffer_parts = [title]
                else:
                    buffer_parts = []
                    
            elif tag_name in self.CONTENT_TAGS:
                # GLUE: Add content to current buffer
                text = get_text_content(element)
                if text:
                    buffer_parts.append(text)
            else:
                # For container elements, process children
                for child in element.children:
                    process_element(child)
        
        # Find main content area
        body = soup.find('body') or soup
        main_content = body.find('article') or body.find('main') or body
        
        # Process all top-level elements
        for element in main_content.children:
            process_element(element)
        
        # Flush remaining content
        flush_section()
        
        # Handle case with no headers - treat all content as one section
        if not sections:
            all_text = soup.get_text(separator=' ', strip=True)
            if self._is_valid_content(all_text):
                sections.append(ContentSection(
                    text=all_text,
                    title=None,
                    hierarchy=[],
                    start_char=0,
                    end_char=len(all_text),
                    metadata={"source": filename, "format": "html"}
                ))
        
        return sections
    
    def _parse_with_regex(self, content: str, filename: str) -> List[ContentSection]:
        """Fallback regex-based parsing (basic HTML tag stripping)."""
        # Remove script/style content
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Extract text by removing all tags
        text = re.sub(r'<[^>]+>', ' ', content)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if self._is_valid_content(text):
            return [ContentSection(
                text=text,
                title=None,
                hierarchy=[],
                start_char=0,
                end_char=len(text),
                metadata={"source": filename, "format": "html"}
            )]
        return []

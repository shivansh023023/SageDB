# HTML Parser - Uses BeautifulSoup for parsing

import re
from typing import List, Optional
from .base import BaseParser, ContentSection

try:
    from bs4 import BeautifulSoup, NavigableString
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class HTMLParser(BaseParser):
    """
    Parser for HTML files.
    
    Uses BeautifulSoup for robust HTML parsing.
    Extracts text from semantic elements while preserving structure.
    
    Key behavior:
    - Strips scripts, styles, and navigation elements
    - Preserves header hierarchy (h1-h6)
    - Only creates sections with actual content
    """
    
    # Tags to completely ignore
    IGNORE_TAGS = {'script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'iframe'}
    
    # Semantic content tags
    CONTENT_TAGS = {'p', 'article', 'section', 'main', 'div', 'span', 'li', 'td', 'th', 'blockquote', 'pre', 'code'}
    
    # Header tags
    HEADER_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    
    def get_file_type(self) -> str:
        return "html"
    
    def parse(self, content: str, filename: str = "") -> List[ContentSection]:
        """Parse HTML content into content-bearing sections."""
        if not content or not content.strip():
            return []
        
        if not BS4_AVAILABLE:
            return self._parse_with_regex(content, filename)
        
        return self._parse_with_beautifulsoup(content, filename)
    
    def _parse_with_beautifulsoup(self, content: str, filename: str) -> List[ContentSection]:
        """Parse using BeautifulSoup."""
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup.find_all(self.IGNORE_TAGS):
            tag.decompose()
        
        sections = []
        current_hierarchy: List[str] = []
        current_content: List[str] = []
        current_title: Optional[str] = None
        
        def process_element(element, hierarchy: List[str]):
            nonlocal current_hierarchy, current_content, current_title, sections
            
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    current_content.append(text)
                return
            
            tag_name = element.name
            if not tag_name:
                return
            
            # Handle headers - they start new sections
            if tag_name in self.HEADER_TAGS:
                # Save previous section if it has content
                if current_content:
                    section_text = ' '.join(current_content).strip()
                    if self._is_valid_content(section_text):
                        sections.append(ContentSection(
                            text=section_text,
                            title=current_title,
                            hierarchy=current_hierarchy.copy(),
                            start_char=0,  # HTML doesn't have clear char positions
                            end_char=0,
                            metadata={"source": filename, "format": "html"}
                        ))
                    current_content = []
                
                # Update hierarchy based on header level
                level = int(tag_name[1])
                title = element.get_text(strip=True)
                
                current_hierarchy = current_hierarchy[:level-1]
                if title:
                    current_hierarchy.append(title)
                current_title = title
                
            elif tag_name in self.CONTENT_TAGS:
                # Get text from content tags
                text = element.get_text(separator=' ', strip=True)
                if text:
                    current_content.append(text)
            else:
                # Recursively process children
                for child in element.children:
                    process_element(child, hierarchy)
        
        # Find the main content area
        body = soup.find('body')
        main_content = body if body else soup
        
        # Try to find article or main element first
        article = main_content.find('article') or main_content.find('main')
        if article:
            main_content = article
        
        # Process all elements
        for element in main_content.children:
            process_element(element, [])
        
        # Don't forget the last section
        if current_content:
            section_text = ' '.join(current_content).strip()
            if self._is_valid_content(section_text):
                sections.append(ContentSection(
                    text=section_text,
                    title=current_title,
                    hierarchy=current_hierarchy.copy(),
                    start_char=0,
                    end_char=0,
                    metadata={"source": filename, "format": "html"}
                ))
        
        # If no sections found, try to get all text
        if not sections:
            all_text = main_content.get_text(separator=' ', strip=True)
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
        """Fallback regex-based parsing when BeautifulSoup is not available."""
        # Remove script and style tags
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all HTML tags
        text = re.sub(r'<[^>]+>', ' ', content)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        if self._is_valid_content(text):
            return [ContentSection(
                text=text.strip(),
                title=None,
                hierarchy=[],
                start_char=0,
                end_char=len(text),
                metadata={"source": filename, "format": "html"}
            )]
        
        return []

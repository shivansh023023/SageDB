# XML Parser - Implements "Sticky Header" pattern for structured data
# Semantic tags act as magnets that hold onto their content

import re
from typing import List, Optional, Dict
from .base import BaseParser, ContentSection

try:
    import xml.etree.ElementTree as ET
    from xml.etree.ElementTree import ParseError
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False


class XMLParser(BaseParser):
    """
    Parser for XML files with Sticky Header pattern.
    
    Key behavior:
    - Semantic parent tags act as MAGNETS
    - Child elements are GLUED to their parent
    - Full XPath provides hierarchy context
    - Attributes are included in section metadata
    
    Example:
        Input:
            <documentation>
                <section title="Installation">
                    <step>Clone the repository</step>
                    <step>Run pip install</step>
                </section>
            </documentation>
        
        Output:
            Section: "Installation: Clone the repository, Run pip install"
            (with hierarchy: ["documentation", "section"])
    """
    
    # Tags that indicate semantic sections (act as magnets)
    SECTION_TAGS = {
        'section', 'chapter', 'article', 'div', 'entry', 'item',
        'record', 'row', 'node', 'element', 'block', 'part',
        'paragraph', 'p', 'header', 'head', 'title', 'topic',
        'doc', 'document', 'page', 'content', 'body', 'main'
    }
    
    # Tags to skip entirely
    SKIP_TAGS = {'comment', 'annotation', 'meta', 'metadata'}
    
    # Attribute names commonly used as titles
    TITLE_ATTRS = ['title', 'name', 'id', 'label', 'heading']
    
    def get_file_type(self) -> str:
        return "xml"
    
    def parse(self, content: str, filename: str = "") -> List[ContentSection]:
        """Parse XML using sticky header pattern."""
        if not content or not content.strip():
            return []
        
        if not XML_AVAILABLE:
            return self._parse_with_regex(content, filename)
        
        try:
            # Remove XML declaration and DOCTYPE if present (can cause issues)
            content = re.sub(r'<\?xml[^?]*\?>', '', content)
            content = re.sub(r'<!DOCTYPE[^>]*>', '', content)
            
            root = ET.fromstring(content)
            sections = []
            self._process_element(root, [], sections, filename)
            return sections
            
        except ParseError:
            return self._parse_with_regex(content, filename)
    
    def _process_element(self, element: 'ET.Element', path: List[str], 
                         sections: List[ContentSection], filename: str) -> None:
        """
        Recursively process XML elements with sticky pattern.
        
        Strategy:
        - If element is a "section tag" - create a section with all child text
        - Otherwise, recurse into children
        """
        tag = element.tag
        
        # Skip certain tags
        if tag.lower() in self.SKIP_TAGS:
            return
        
        # Update path
        current_path = path + [tag]
        
        # Get title from common attributes
        title = None
        for attr in self.TITLE_ATTRS:
            if attr in element.attrib:
                title = element.attrib[attr]
                break
        
        # Check if this is a section-level element
        is_section = (
            tag.lower() in self.SECTION_TAGS or
            title is not None or
            self._has_direct_text(element)
        )
        
        if is_section:
            # STICKY: Collect all text under this element as one section
            text_content = self._collect_text(element, include_tag_title=True)
            
            if text_content and self._is_valid_content(text_content):
                # Build section text with title if available
                if title:
                    section_text = f"{title}: {text_content}"
                else:
                    section_text = text_content
                
                sections.append(ContentSection(
                    text=section_text,
                    title=title or tag,
                    hierarchy=current_path.copy(),
                    start_char=0,
                    end_char=len(section_text),
                    metadata={
                        "source": filename,
                        "format": "xml",
                        "xpath": '/' + '/'.join(current_path),
                        "tag": tag,
                        "attributes": dict(element.attrib) if element.attrib else None
                    }
                ))
        
        # Always process children for deeper sections
        for child in element:
            self._process_element(child, current_path, sections, filename)
    
    def _has_direct_text(self, element: 'ET.Element') -> bool:
        """Check if element has direct text content (not just in children)."""
        if element.text and element.text.strip():
            return True
        return False
    
    def _collect_text(self, element: 'ET.Element', include_tag_title: bool = False) -> str:
        """Collect all text from element and descendants."""
        parts = []
        
        # Direct text
        if element.text and element.text.strip():
            parts.append(element.text.strip())
        
        # Child elements' text
        for child in element:
            child_text = self._collect_text(child, include_tag_title=False)
            if child_text:
                parts.append(child_text)
            # Tail text (text after child tag)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())
        
        return ' '.join(parts)
    
    def _parse_with_regex(self, content: str, filename: str) -> List[ContentSection]:
        """Fallback regex-based parsing for invalid XML."""
        sections = []
        
        # Try to find tag + content patterns
        # Pattern: <tag...>content</tag>
        pattern = re.compile(r'<(\w+)[^>]*>(.*?)</\1>', re.DOTALL)
        
        for match in pattern.finditer(content):
            tag = match.group(1)
            inner = match.group(2)
            
            # Skip if tag is in skip list
            if tag.lower() in self.SKIP_TAGS:
                continue
            
            # Clean inner content (remove nested tags)
            text = re.sub(r'<[^>]+>', ' ', inner)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if self._is_valid_content(text):
                sections.append(ContentSection(
                    text=f"{tag}: {text}" if tag else text,
                    title=tag,
                    hierarchy=[tag],
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={
                        "source": filename,
                        "format": "xml",
                        "tag": tag,
                        "regex_fallback": True
                    }
                ))
        
        # If no patterns found, extract all text
        if not sections:
            all_text = re.sub(r'<[^>]+>', ' ', content)
            all_text = re.sub(r'\s+', ' ', all_text).strip()
            
            if self._is_valid_content(all_text):
                sections.append(ContentSection(
                    text=all_text,
                    title=None,
                    hierarchy=[],
                    start_char=0,
                    end_char=len(content),
                    metadata={"source": filename, "format": "xml"}
                ))
        
        return sections

# JSON Parser - For structured JSON files

import json
from typing import List, Any, Dict, Optional
from .base import BaseParser, ContentSection


class JSONParser(BaseParser):
    """
    Parser for JSON files.
    
    Recursively traverses JSON structure and extracts text content.
    Preserves hierarchy based on nesting level.
    
    Key behavior:
    - Looks for text-like fields: "text", "content", "description", "body", "value"
    - Nested objects create hierarchy
    - Arrays are flattened with index in metadata
    """
    
    # Fields that likely contain text content
    TEXT_FIELDS = {'text', 'content', 'description', 'body', 'value', 'message', 
                   'summary', 'title', 'name', 'label', 'data', 'info'}
    
    # Fields that indicate a title/name
    TITLE_FIELDS = {'title', 'name', 'label', 'id', 'key', 'heading'}
    
    def get_file_type(self) -> str:
        return "json"
    
    def parse(self, content: str, filename: str = "") -> List[ContentSection]:
        """Parse JSON content into sections."""
        if not content or not content.strip():
            return []
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Return empty if invalid JSON
            return []
        
        sections = []
        self._extract_sections(data, [], sections, filename)
        return sections
    
    def _extract_sections(self, data: Any, hierarchy: List[str], 
                         sections: List[ContentSection], filename: str,
                         parent_title: Optional[str] = None):
        """Recursively extract sections from JSON data."""
        
        if isinstance(data, dict):
            # Try to find a title for this object
            title = None
            for field in self.TITLE_FIELDS:
                if field in data and isinstance(data[field], str):
                    title = data[field]
                    break
            
            # Collect text content from this object
            text_parts = []
            for key, value in data.items():
                if key.lower() in self.TEXT_FIELDS and isinstance(value, str):
                    if self._is_valid_content(value):
                        text_parts.append(value)
                elif isinstance(value, (dict, list)):
                    # Recurse into nested structures
                    new_hierarchy = hierarchy.copy()
                    if title:
                        new_hierarchy.append(title)
                    self._extract_sections(value, new_hierarchy, sections, filename, title)
            
            # Create section if we found text content
            if text_parts:
                combined_text = '\n'.join(text_parts)
                if self._is_valid_content(combined_text):
                    section_hierarchy = hierarchy.copy()
                    if title and title not in section_hierarchy:
                        section_hierarchy.append(title)
                    
                    sections.append(ContentSection(
                        text=combined_text,
                        title=title,
                        hierarchy=section_hierarchy,
                        start_char=0,
                        end_char=len(combined_text),
                        metadata={"source": filename, "format": "json"}
                    ))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._extract_sections(item, hierarchy, sections, filename, 
                                      f"Item {i}" if parent_title is None else parent_title)
        
        elif isinstance(data, str):
            # Plain string at current level
            if self._is_valid_content(data):
                sections.append(ContentSection(
                    text=data,
                    title=parent_title,
                    hierarchy=hierarchy.copy(),
                    start_char=0,
                    end_char=len(data),
                    metadata={"source": filename, "format": "json"}
                ))

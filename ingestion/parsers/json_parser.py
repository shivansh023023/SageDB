# JSON Parser - Implements "Sticky Header" pattern for structured data
# Keys act as magnets that hold onto their values (especially arrays)

import json
import re
from typing import Any, List, Optional
from .base import BaseParser, ContentSection


class JSONParser(BaseParser):
    """
    Parser for JSON files with Sticky Header pattern.
    
    Key behavior:
    - Object keys act as MAGNETS for their values
    - Array values are GLUED together with their key
    - Nested structures preserve full path context
    - NEVER separate a key from its list/array values
    
    Example:
        Input:
            {
                "priorities": ["ingestion", "storage", "retrieval"],
                "deadline": "January 15"
            }
        
        Output:
            Section 1: "priorities: ingestion, storage, retrieval"
            Section 2: "deadline: January 15"
        
        NOT:
            Section 1: "priorities"
            Section 2: "ingestion, storage, retrieval"  <- WRONG! Separates key from value
    """
    
    def get_file_type(self) -> str:
        return "json"
    
    def parse(self, content: str, filename: str = "") -> List[ContentSection]:
        """Parse JSON using sticky header pattern."""
        if not content or not content.strip():
            return []
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract whatever we can
            return self._parse_invalid_json(content, filename)
        
        sections = []
        self._flatten_json(data, [], sections, filename)
        return sections
    
    def _flatten_json(self, data: Any, path: List[str], sections: List[ContentSection], 
                      filename: str, depth: int = 0) -> None:
        """
        Recursively flatten JSON with sticky key-value pairs.
        
        Strategy:
        - Scalar values: key + value as one section
        - Arrays: key + ALL array elements as one section
        - Nested objects: recurse with path context
        """
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = path + [str(key)]
                
                if isinstance(value, dict):
                    # Recurse into nested object
                    self._flatten_json(value, current_path, sections, filename, depth + 1)
                elif isinstance(value, list):
                    # STICKY: Key + entire array as one section
                    self._add_array_section(key, value, current_path, sections, filename)
                else:
                    # STICKY: Key + scalar value as one section
                    self._add_scalar_section(key, value, current_path, sections, filename)
                    
        elif isinstance(data, list):
            # Top-level array
            if len(data) > 0:
                if all(isinstance(item, dict) for item in data):
                    # Array of objects - process each
                    for i, item in enumerate(data):
                        item_path = path + [f"[{i}]"]
                        self._flatten_json(item, item_path, sections, filename, depth + 1)
                else:
                    # Array of primitives or mixed
                    self._add_array_section("items", data, path, sections, filename)
        else:
            # Top-level primitive (rare but valid JSON)
            if self._is_valid_content(str(data)):
                sections.append(ContentSection(
                    text=str(data),
                    title=None,
                    hierarchy=path,
                    start_char=0,
                    end_char=len(str(data)),
                    metadata={"source": filename, "format": "json"}
                ))
    
    def _add_scalar_section(self, key: str, value: Any, path: List[str], 
                            sections: List[ContentSection], filename: str) -> None:
        """Add a key-value pair as a section (STICKY: key + value together)."""
        # Format: "key: value"
        value_str = self._format_value(value)
        section_text = f"{key}: {value_str}"
        
        if self._is_valid_content(section_text):
            sections.append(ContentSection(
                text=section_text,
                title=key,
                hierarchy=path.copy(),
                start_char=0,
                end_char=len(section_text),
                metadata={
                    "source": filename, 
                    "format": "json",
                    "json_path": '.'.join(path),
                    "value_type": type(value).__name__
                }
            ))
    
    def _add_array_section(self, key: str, array: List[Any], path: List[str], 
                           sections: List[ContentSection], filename: str) -> None:
        """Add key + entire array as ONE section (NEVER split)."""
        if not array:
            return
        
        # Format array elements
        if all(isinstance(item, (str, int, float, bool)) for item in array):
            # Simple array - join as comma-separated or bullet list
            if len(array) <= 5:
                # Short list: comma-separated
                items_str = ', '.join(str(item) for item in array)
                section_text = f"{key}: {items_str}"
            else:
                # Long list: bullet points
                items_str = '\n'.join(f"• {item}" for item in array)
                section_text = f"{key}:\n{items_str}"
        elif all(isinstance(item, dict) for item in array):
            # Array of objects - format each object
            parts = [f"{key}:"]
            for i, obj in enumerate(array):
                obj_str = self._format_object(obj)
                parts.append(f"  [{i + 1}] {obj_str}")
            section_text = '\n'.join(parts)
        else:
            # Mixed array
            items_str = ', '.join(self._format_value(item) for item in array)
            section_text = f"{key}: {items_str}"
        
        if self._is_valid_content(section_text):
            sections.append(ContentSection(
                text=section_text,
                title=key,
                hierarchy=path.copy(),
                start_char=0,
                end_char=len(section_text),
                metadata={
                    "source": filename,
                    "format": "json",
                    "json_path": '.'.join(path),
                    "value_type": "array",
                    "array_length": len(array)
                }
            ))
    
    def _format_value(self, value: Any) -> str:
        """Format a JSON value as readable string."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            return ', '.join(self._format_value(v) for v in value)
        elif isinstance(value, dict):
            return self._format_object(value)
        return str(value)
    
    def _format_object(self, obj: dict) -> str:
        """Format a dict object as readable string."""
        if not obj:
            return "{}"
        parts = []
        for k, v in obj.items():
            parts.append(f"{k}: {self._format_value(v)}")
        return '; '.join(parts)
    
    def _parse_invalid_json(self, content: str, filename: str) -> List[ContentSection]:
        """
        Handle invalid JSON by extracting key-value patterns.
        Useful for JSON-like config files with comments.
        """
        # Try to extract "key": "value" or "key": [...] patterns
        sections = []
        
        # Pattern for "key": value (handles strings, numbers, arrays, objects)
        pattern = re.compile(r'"([^"]+)"\s*:\s*(\[[^\]]*\]|"[^"]*"|\d+\.?\d*|true|false|null)')
        
        for match in pattern.finditer(content):
            key = match.group(1)
            value = match.group(2)
            
            # Clean up value
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith('[') and value.endswith(']'):
                # Try to parse array
                try:
                    arr = json.loads(value)
                    if isinstance(arr, list):
                        value = ', '.join(str(v) for v in arr)
                except:
                    value = value[1:-1]  # Strip brackets
            
            section_text = f"{key}: {value}"
            if self._is_valid_content(section_text):
                sections.append(ContentSection(
                    text=section_text,
                    title=key,
                    hierarchy=[key],
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={"source": filename, "format": "json", "partial_parse": True}
                ))
        
        # If no patterns found, treat as plain text
        if not sections and self._is_valid_content(content.strip()):
            sections.append(ContentSection(
                text=content.strip(),
                title=None,
                hierarchy=[],
                start_char=0,
                end_char=len(content),
                metadata={"source": filename, "format": "json", "parse_failed": True}
            ))
        
        return sections

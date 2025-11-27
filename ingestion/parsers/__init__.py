# File Parsers Module
# Each parser extracts content-bearing sections from different file types
# All parsers implement "Sticky Header" pattern: headers are magnets that hold content

from .base import BaseParser, ContentSection
from .markdown_parser import MarkdownParser
from .html_parser import HTMLParser
from .text_parser import TextParser
from .json_parser import JSONParser
from .xml_parser import XMLParser

__all__ = [
    'BaseParser',
    'ContentSection', 
    'MarkdownParser',
    'HTMLParser',
    'TextParser',
    'JSONParser',
    'XMLParser',
]

# Parser registry - maps file type to parser class
PARSER_REGISTRY = {
    'markdown': MarkdownParser,
    'html': HTMLParser,
    'text': TextParser,
    'json': JSONParser,
    'xml': XMLParser,
}


def get_parser(file_type: str) -> BaseParser:
    """Get the appropriate parser for a file type."""
    parser_class = PARSER_REGISTRY.get(file_type)
    if parser_class is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    return parser_class()

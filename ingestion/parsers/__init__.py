# File Parsers Module
# Each parser extracts content-bearing sections from different file types

from .base import BaseParser, ContentSection
from .markdown_parser import MarkdownParser
from .html_parser import HTMLParser
from .text_parser import TextParser
from .json_parser import JSONParser

__all__ = [
    'BaseParser',
    'ContentSection', 
    'MarkdownParser',
    'HTMLParser',
    'TextParser',
    'JSONParser',
]

# Parser registry - maps file type to parser class
PARSER_REGISTRY = {
    'markdown': MarkdownParser,
    'html': HTMLParser,
    'text': TextParser,
    'json': JSONParser,
}


def get_parser(file_type: str) -> BaseParser:
    """Get the appropriate parser for a file type."""
    parser_class = PARSER_REGISTRY.get(file_type)
    if parser_class is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    return parser_class()

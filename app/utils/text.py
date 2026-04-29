import re

import bleach
import markdown

# Allowed HTML tags for LLM response sanitization
_ALLOWED_TAGS = [
    'p', 'br', 'b', 'i', 'strong', 'em',
    'ul', 'ol', 'li',
    'h1', 'h2', 'h3', 'h4',
    'pre', 'code', 'blockquote',
    'table', 'thead', 'tbody', 'tr', 'th', 'td',
]


def render_markdown_safe(text: str) -> str:
    """Convert markdown to sanitized HTML (no attributes allowed)."""
    html = markdown.markdown(text, extensions=['extra', 'nl2br'])
    return bleach.clean(html, tags=_ALLOWED_TAGS, attributes={}, strip=True)


def clean_organism_name(text: str) -> str:
    """
    Strip numeric characters and bracketed substrings from organism names.
    E.g. 'Mycobacterium (ATCC 1234)' -> 'Mycobacterium'
    """
    text = re.sub(r'\d+', '', str(text))
    while '(' in text and ')' in text:
        text = re.sub(r'\([^()]*\)', '', text)
    return text.strip()

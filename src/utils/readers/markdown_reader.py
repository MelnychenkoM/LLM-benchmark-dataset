from pathlib import Path
from typing import Any, Union

class MarkdownReader:
    """ Dummy class for md reading """
    def __call__(self, md_path: str) -> str:
        with open(md_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.readlines()
        text = "\n".join(line for line in text)
        return text
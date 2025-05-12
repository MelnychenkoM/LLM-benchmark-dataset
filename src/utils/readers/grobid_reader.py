import scipdf
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Set, List, Tuple, Union


class PDFGrobidReader:
    """
    Parses a PDF article using GROBID (via scipdf) and formats its
    content (Title, Abstract, Sections, Figure Captions) into a
    single Markdown string.

    Usage:
        reader = PDFGrobidReader()
        markdown_text = reader("path/to/your/article.pdf")
    """

    @staticmethod
    def _add_section_ids(article_dict: Dict[str, Any]):
        """Adds a unique 'section_id' to each section dictionary. (Internal helper)"""
        if 'sections' not in article_dict or not isinstance(article_dict.get('sections'), list):
            article_dict['sections'] = []
            return

        for idx, section in enumerate(article_dict['sections']):
            if isinstance(section, dict):
                section['section_id'] = idx

    @staticmethod
    def _get_figure_to_section_mapping(article_dict: Dict[str, Any]) -> Dict[str, int]:
        """
        Determines the most likely section for each figure based on reference frequency.
        Returns a dictionary mapping figure_id to section_id. (Internal helper)
        """
        section_figures: List[Tuple[int, Counter]] = []
        if 'sections' not in article_dict or not isinstance(article_dict.get('sections'), list):
            return {}

        for section in article_dict['sections']:
            if isinstance(section, dict) and 'section_id' in section:
                fig_refs = section.get('figure_ref', [])
                if isinstance(fig_refs, list):
                    section_figures.append((section['section_id'], Counter(fig_refs)))

        sorted_sections = sorted(section_figures, key=lambda x: sum(x[1].values()), reverse=True)

        figure_to_section: Dict[str, int] = {}
        assigned_figures: Set[str] = set()

        for section_id, fig_counter in sorted_sections:
            for fig_id, count in fig_counter.items():
                if fig_id not in assigned_figures:
                    figure_to_section[fig_id] = section_id
                    assigned_figures.add(fig_id)

        return figure_to_section

    @staticmethod
    def _format_article_to_md_string(article_dict: Dict[str, Any], figure_to_section: Dict[str, int]) -> str:
        """
        Formats the parsed article dictionary into a Markdown string. (Internal helper)
        """
        if not article_dict or not isinstance(article_dict, dict):
            return ""

        md_parts: List[str] = []
        written_figures: Set[str] = set()

        md_parts.append(f"# Title\n{article_dict.get('title', 'N/A')}\n")
        md_parts.append(f"# Abstract\n{article_dict.get('abstract', 'N/A')}\n")

        for section in article_dict.get("sections", []):
            if not isinstance(section, dict): continue

            heading = section.get('heading')
            text = section.get('text', '')
            section_id = section.get('section_id')

            if heading:
                md_parts.append(f"# {heading}\n{text}\n")
            else:
                md_parts.append(f"{text}\n\n")

            fig_refs = section.get('figure_ref', [])
            if section_id is not None and isinstance(fig_refs, list):
                for fig_id in fig_refs:
                    if figure_to_section.get(fig_id) == section_id and fig_id not in written_figures:
                        for fig in article_dict.get('figures', []):
                            if isinstance(fig, dict) and fig.get('figure_id') == fig_id:
                                label = fig.get('figure_label', 'N/A')
                                caption = fig.get('figure_caption', 'N/A')
                                md_parts.append(f"## Figure {label} Caption\n{caption}\n")
                                written_figures.add(fig_id)
                                break

        return "".join(md_parts)

    def __call__(self, article_path: Union[str, Path]) -> str:
        """
        Parses the PDF file at the given path and returns its content
        as a formatted Markdown string.

        Args:
            article_path: The file path (string or Path object) to the PDF article.

        Returns:
            A string containing the formatted article content (Title, Abstract,
            Sections, Figure Captions), or an error message string starting
            with "Error:" if parsing or processing fails.
        """
        try:
            article_path_str = str(article_path)
            if not Path(article_path_str).is_file():
                 return f"Error: File not found at {article_path_str}"

            article_dict = scipdf.parse_pdf_to_dict(article_path_str, as_list=False)

            if not article_dict or not isinstance(article_dict, dict):
                return f"Error: Failed to parse or received invalid dictionary for {article_path_str}"

            self._add_section_ids(article_dict)

            figure_to_section = self._get_figure_to_section_mapping(article_dict)
            markdown_text = self._format_article_to_md_string(article_dict, figure_to_section)

            return markdown_text

        except FileNotFoundError:
             return f"Error: File not found at {article_path_str}"
        except Exception as e:
            return f"Error processing {article_path}: {type(e).__name__} - {e}"
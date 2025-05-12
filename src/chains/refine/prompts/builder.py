from chains.prompt_buider import PromptBuilder, ArtifactDirFile
from .templates import (
    SYSTEM_MESSAGE,
    INITIAL_CALL_PROMPT,
    REFINE_CALL_PROMPT,
    EXTRACTION_INSTRUCTIONS,
    FORMAT_INSTRUCTIONS,
    RETRY_CORRECTION_PROMPT,
    SYSTEM_MESSAGE_FILTER,
    FILTER_PAPERS_PROMPT
)


class RefineChainPromptsFromArtifactsBuilder(PromptBuilder):
    @property
    def ARTIFACT_NAME(self) -> str:
        return "refine_chain_prompts"

    @property
    def ARTIFACT_DIR(self) -> list[ArtifactDirFile]:
        return [
            {
                'prompt_name': 'SYSTEM_MESSAGE',
                'prompt': SYSTEM_MESSAGE,
                'subfolders': ['system'],
                'file_name': 'message'
            },
            {
                'prompt_name': 'SYSTEM_MESSAGE_FILTER',
                'prompt': SYSTEM_MESSAGE_FILTER,
                'subfolders': ['system'],
                'file_name': 'message_filter'
            },
            {
                'prompt_name': 'INITIAL_CALL_PROMPT',
                'prompt': INITIAL_CALL_PROMPT,
                'subfolders': ['human'],
                'file_name': 'initial_call'
            },
            {
                'prompt_name': 'FILTER_PAPERS_PROMPT',
                'prompt': FILTER_PAPERS_PROMPT,
                'subfolders': ['human'],
                'file_name': 'filter_papers'
            },
            {
                'prompt_name': 'REFINE_CALL_PROMPT',
                'prompt': REFINE_CALL_PROMPT,
                'subfolders': ['human'],
                'file_name': 'refine_call'
            },
            {
                'prompt_name': 'RETRY_CORRECTION_PROMPT',
                'prompt': RETRY_CORRECTION_PROMPT,
                'subfolders': ['human'],
                'file_name': 'retry_correction_call'
            },
            {
                'prompt_name': 'EXTRACTION_INSTRUCTIONS',
                'prompt': EXTRACTION_INSTRUCTIONS,
                'subfolders': ['human', 'instructions'],
                'file_name': 'extract'
            },
            {
                'prompt_name': 'FORMAT_INSTRUCTIONS',
                'prompt': FORMAT_INSTRUCTIONS,
                'subfolders': ['human', 'instructions'],
                'file_name': 'parse_format'
            },
        ]

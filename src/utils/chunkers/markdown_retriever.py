from langchain.docstore.document import Document
import tiktoken

class MarkdownChunker:
    """
    Retrieve .md text in one chunk.
    """
    def __init__(self,
                chunk_size: int,
                openai_model_name: str = "gpt-4.0-mini",
                return_as_langchain_documents: bool = True
                ) -> None:
        self.openai_model_name = openai_model_name
        self.return_as_langchain_documents = return_as_langchain_documents
        self.tokenizer = None

    def __call__(self, text: str):
        if self.return_as_langchain_documents:
            return [Document(page_content=text, metadata={'chunk_id': 1})]
        return text
import tiktoken
from langchain.docstore.document import Document


class CustomOpenAIChunker():
    """
      Split a text into smaller chunks of size n, preferably ending at the end of
      a sentence. Thus the actual size of chunks may not be fixed.
    """

    def __init__(self,
                 chunk_size: int = 1000,
                 openai_model_name: str = "gpt-3.5-turbo",
                 return_as_langchain_documents: bool = True):
        self.parameters = {
            'chunker': self.__class__.__name__,
            'chunk_size': chunk_size
        }
        self.tokenizer = tiktoken.encoding_for_model(openai_model_name)
        self.chunk_size = chunk_size
        self.return_as_langchain_documents = return_as_langchain_documents

    def __call__(self, text: str):
        chunks = [self.tokenizer.decode(chunk)
                  for chunk in self._create_chunks(text)]

        if self.return_as_langchain_documents:
            return [Document(page_content=chunk, metadata={'chunk_id': i}) for i, chunk in enumerate(chunks)]
        else:
            return chunks

    def _create_chunks(self, text):
        tokenizer = self.tokenizer
        chunk_size = self.chunk_size

        tokens = tokenizer.encode(text)
        """Yield successive n-sized chunks from text."""
        i = 0
        while i < len(tokens):
            # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
            j = min(i + int(1.5 * chunk_size), len(tokens))
            while j > i + int(0.5 * chunk_size):
                # Decode the tokens and check for full stop or newline
                chunk = tokenizer.decode(tokens[i:j])
                if chunk.endswith(".") or chunk.endswith("\n"):
                    break
                j -= 1
            # If no end of sentence found, use n tokens as the chunk size
            if j == i + int(0.5 * chunk_size):
                j = min(i + chunk_size, len(tokens))
            yield tokens[i:j]
            i = j
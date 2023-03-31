import itertools
from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma
from pydantic import BaseModel, Extra, Field


class InputText(BaseModel):
    text: str = Field(title="Text", description="Text to split into chunks.", min_length=1)
    metadata: dict[str, str] = Field(title="Metadata", description="Metadata associated with the text.")


class PersistentVectorStore(BaseModel):
    """Store for a database."""

    text_splitter: TextSplitter = Field(
        title="Text Splitter",
        description="Text Splitter to use to split documents into chunks.",
    )
    chroma: Chroma = Field(
        title="Chroma Vector Store",
        description="Vector Store to use to store the documents.",
    )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def store(self, input_texts: List[InputText]) -> None:
        chunked_documents = list(
            itertools.chain.from_iterable([self._convert_text_to_documents(input_text) for input_text in input_texts])
        )
        self.chroma.add_documents(chunked_documents)
        self.chroma.persist()

    def _convert_text_to_documents(self, input_text: InputText) -> List[Document]:
        texts = self.text_splitter.split_text(input_text.text)
        return [Document(page_content=text, metadata=input_text.metadata) for text in texts]

from pathlib import Path

import click
import structlog
from google.cloud import bigquery
from langchain.callbacks import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from chatgpt_playground.chains.bigquery import BigQuerySQLSequentialChain

from .cli import cli

logger = structlog.get_logger(__name__)


@cli.command(help="List tables of BigQuery for query")
@click.option(
    "--query",
    required=True,
    help="Query for LLM",
)
@click.option(
    "--persistent-directory",
    help="Directory to store persistent data.",
    default=Path("./chroma"),
    type=click.Path(writable=True, path_type=Path),
)
def query_sql(query: str, persistent_directory: Path) -> None:
    """Query metadata from Chroma."""
    embeddings = OpenAIEmbeddings()
    chroma = Chroma(persist_directory=str(persistent_directory), embedding_function=embeddings)

    llm = ChatOpenAI(
        streaming=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
    )
    chain = BigQuerySQLSequentialChain.from_deps(
        llm=llm,
        chroma=chroma,
        client=bigquery.Client(),
    )

    query = chain.run(query=query)
    logger.info("Query", query=query)

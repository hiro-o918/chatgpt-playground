from pathlib import Path

import click
import structlog
from google.cloud.bigquery import Client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from chatgpt_playground.bigquery import list_table_metadata_list_of_dataset
from chatgpt_playground.vector_store import InputText, PersistentVectorStore

from .cli import cli

logger = structlog.get_logger(__name__)


@cli.command(help="Load metadata from BigQuery.")
@click.option("--project-id", required=True, help="Project ID (e.g. 'my_project')")
@click.option("--dataset-id", required=True, help="Dataset ID (e.g. 'my_dataset')")
@click.option(
    "--persistent-directory",
    help="Directory to store persistent data.",
    default=Path("./chroma"),
    type=click.Path(writable=True, path_type=Path),
)
def load_bigquery_metadata(project_id: str, dataset_id: str, persistent_directory: Path) -> None:
    """Load metadata from BigQuery."""
    client = Client()
    logger.info("Loading metadata from BigQuery...")
    tables = [
        InputText(
            text=t.json(),
            metadata={
                "source": t.id,
                "id": t.id,
                "project_id": t.project_id,
                "dataset_id": t.dataset_id,
                "table_id": t.table_id,
            },
        )
        for t in list_table_metadata_list_of_dataset(client, project_id, dataset_id)
    ]
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100, separator=",")
    embeddings = OpenAIEmbeddings()
    chroma = Chroma(persist_directory=str(persistent_directory), embedding_function=embeddings)
    vector_store = PersistentVectorStore(
        text_splitter=text_splitter,
        chroma=chroma,
    )
    logger.info("Storing metadata in Chroma...")
    vector_store.store(tables)

from typing import Generator

from google.cloud import bigquery
from pydantic import BaseModel, Field


class TableSchemaField(BaseModel):
    name: str = Field(title="Name", description="Name")
    type: str = Field(title="Type", description="Type")


class TableIdentifier(BaseModel):
    project_id: str = Field(title="Project ID", description="Project ID")
    dataset_id: str = Field(title="Dataset ID", description="Dataset ID")
    table_id: str = Field(title="Table ID", description="Table ID")


class TableMetadata(TableIdentifier):
    id: str = Field(title="ID", description="ID")
    description: str | None = Field(title="Description", description="Description", default=None)
    table_schema: list[TableSchemaField] = Field(
        title="Schema",
        description="Schema",
    )


def list_table_metadata_list_of_dataset(
    client: bigquery.Client, project_id: str, dataset_id: str
) -> Generator[TableMetadata, None, None]:
    dataset_ref = client.dataset(dataset_id, project=project_id)
    tables = client.list_tables(dataset_ref)

    for table_ref in tables:
        table = client.get_table(table_ref)
        schemas = [
            TableSchemaField(
                name=field.name,
                type=field.field_type,
            )
            for field in table.schema
        ]
        table_metadata = TableMetadata(
            id=f"{project_id}.{dataset_id}.{table.table_id}",
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table.table_id,
            table_schema=schemas,
        )
        yield table_metadata


def get_table_metadata(
    client: bigquery.Client,
    table_identifier: TableIdentifier,
) -> TableMetadata:
    id_ = f"{table_identifier.project_id}.{table_identifier.dataset_id}.{table_identifier.table_id}"
    table = client.get_table(id_)
    schemas = [
        TableSchemaField(
            name=field.name,
            type=field.field_type,
        )
        for field in table.schema
    ]
    table_metadata = TableMetadata(
        id=id_,
        project_id=table_identifier.project_id,
        dataset_id=table_identifier.dataset_id,
        table_id=table_identifier.table_id,
        table_schema=schemas,
    )
    return table_metadata

from typing import Any

import structlog
from google.cloud import bigquery
from langchain import BasePromptTemplate, LLMChain, PromptTemplate
from langchain.chains.base import Chain
from langchain.schema import BaseLanguageModel
from langchain.vectorstores import Chroma
from pydantic import BaseModel, Extra, Field

from chatgpt_playground.bigquery import TableIdentifier, get_table_metadata

logger = structlog.get_logger(__name__)


_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

The table name must be qualified by `project_id`.`dataset_id`.`table_id`. You must use the full name of the table.

Input language is {input_language}.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the tables listed below.

{table_info}

Question: {input}"""

PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "input_language"],
    template=_DEFAULT_TEMPLATE,
)


class BigQueryTableDeciderChain(Chain, BaseModel):
    """Chain to decide which table to use for a given query."""

    client: bigquery.Client
    top_k: int = Field(
        title="Top K",
        description="Number of results to return.",
    )
    chroma: Chroma = Field(
        title="Chroma Vector Store",
        description="Vector Store to use to store the documents.",
    )

    input_key: str = "query"  #: :meta private:
    output_key: str = "table_metadata_list"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _call(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.callback_manager.on_text(inputs[self.input_key])
        documents = self.chroma.similarity_search(
            query=inputs[self.input_key],
            k=self.top_k,
        )
        # filter out duplicates
        table_metadata_map = {}
        for document in documents:
            table_metadata_map[document.metadata["id"]] = get_table_metadata(
                self.client,
                table_identifier=TableIdentifier(
                    project_id=document.metadata["project_id"],
                    dataset_id=document.metadata["dataset_id"],
                    table_id=document.metadata["table_id"],
                ),
            )
        table_ids = list(table_metadata_map.keys())
        logger.info("Using the following tables:\n%s", table_ids)
        return {self.output_key: list(table_metadata_map.values())}

    @property
    def input_keys(self) -> list[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        return "big_query_table_decider_chain"


class BigQueryTableSQLGeneratorChain(Chain, BaseModel):
    llm: BaseLanguageModel
    client: bigquery.Client

    prompt: BasePromptTemplate = PROMPT

    input_key_table_metadata_list: str = "table_metadata_list"  #: :meta private:
    input_key_query: str = "query"  #: :meta private:
    output_key: str = "sql"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _call(self, inputs: dict[str, Any]) -> dict[str, Any]:
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        table_metadata_list = inputs[self.input_key_table_metadata_list]
        input_text = f"{inputs[self.input_key_query]} \nSQLQuery:"
        llm_inputs = {
            "input": input_text,
            "dialect": "bigquery",
            "table_info": "\n".join([table_metadata.json() for table_metadata in table_metadata_list]),
            "input_language": "Japanese",
            "stop": ["\nSQLResult:"],
        }
        sql = llm_chain.predict(**llm_inputs)
        return {self.output_key: sql}

    @property
    def input_keys(self) -> list[str]:
        return [self.input_key_table_metadata_list, self.input_key_query]

    @property
    def output_keys(self) -> list[str]:
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        return "big_query_table_sql_generator_chain"


class BigQuerySQLSequentialChain(Chain, BaseModel):
    decider_chain: BigQueryTableDeciderChain
    generator_chain: BigQueryTableSQLGeneratorChain
    input_key: str = "query"
    output_key: str = "sql"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @classmethod
    def from_deps(
        cls,
        llm: BaseLanguageModel,
        client: bigquery.Client,
        chroma: Chroma,
        top_k_table_candidates: int = 5,
        **kwargs: Any,
    ) -> "BigQuerySQLSequentialChain":
        decider_chain = BigQueryTableDeciderChain(
            client=client,
            top_k=top_k_table_candidates,
            chroma=chroma,
        )
        generator_chain = BigQueryTableSQLGeneratorChain(
            llm=llm,
            client=client,
        )
        return cls(
            decider_chain=decider_chain,
            generator_chain=generator_chain,
            **kwargs,
        )

    @property
    def input_keys(self) -> list[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        return [self.output_key]

    def _call(self, inputs: dict[str, Any]) -> dict[str, Any]:
        decider_inputs = {input_key: inputs[input_key] for input_key in self.decider_chain.input_keys}
        decider_outputs = self.decider_chain(decider_inputs)

        inputs2 = {**inputs, **decider_outputs}
        generator_inputs = {input_key: inputs2[input_key] for input_key in self.generator_chain.input_keys}
        generator_outputs = self.generator_chain(generator_inputs)

        return {output_key: generator_outputs[output_key] for output_key in self.output_keys}

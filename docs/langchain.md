# langchain

[langchain](https://github.com/hwchase17/langchain) を触る上で気になったことをまとめる。

## SQL 生成方法から学ぶ Chain の体系

[SQL Database Agent](https://python.langchain.com/en/latest/modules/agents/toolkits/examples/sql_database.html?highlight=sql) では SQLLite に対して、SQLAlchemy を使って操作する例が書かれている。

上記の例では [`SQLDatabaseChain`](https://github.com/hwchase17/langchain/blob/65c0c73597d93f59c4afb47a398737038f081153/langchain/chains/sql_database/base.py#L16) を利用し、これが LLM と SQL の間でやり取りを行う。

一方、[`SQLDatabaseSequentialChain`](https://github.com/hwchase17/langchain/blob/65c0c73597d93f59c4afb47a398737038f081153/langchain/chains/sql_database/base.py#L110) というものも存在しており、対象の table が多い場合はこちらのほうが適切であると記述がある。

よって `SQLDatabaseSequentialChain` を中心に中身を見ていく。

### Chain とは

[Getting Started](https://python.langchain.com/en/latest/modules/chains/getting_started.html) に記述されている通り、`Chain` は LLM との prompt のやりとりを記述するためのものである。
単一のやり取りをするものを `Chain`、複数回のやりとりを行うものを `SequentialChain` と呼んでいる。

抽象クラスとしての [Chain](https://github.com/hwchase17/langchain/blob/65c0c73597d93f59c4afb47a398737038f081153/langchain/chains/base.py#L20-L282) で実装されている。
よって、自作する Chain は上記の抽象クラスを継承し自身で実装をする必要がある。

### SQLDatabaseSequentialChain

それでは [SQLDatabaseSequentialChain](https://github.com/hwchase17/langchain/blob/65c0c73597d93f59c4afb47a398737038f081153/langchain/chains/sql_database/base.py#L164-L182) の中身を見ていく。
下記は上記のリンクを引用したものである。

```python
def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
    _table_names = self.sql_chain.database.get_table_names()
    table_names = ", ".join(_table_names)
    llm_inputs = {
        "query": inputs[self.input_key],
        "table_names": table_names,
    }
    table_names_to_use = self.decider_chain.predict_and_parse(**llm_inputs)
    self.callback_manager.on_text(
        "Table names to use:", end="\n", verbose=self.verbose
    )
    self.callback_manager.on_text(
        str(table_names_to_use), color="yellow", verbose=self.verbose
    )
    new_inputs = {
        self.sql_chain.input_key: inputs[self.input_key],
        "table_names_to_use": table_names_to_use,
    }
    return self.sql_chain(new_inputs, return_only_outputs=True)
```

- `self.sql_chain.database.get_table_names()` で対象の DB の table 名一覧を取得する
- `llm_inputs` にはユーザーが入力したクエリと、DB の table 名一覧を格納し、LLM に対して利用すべき table 一覧を取得する
- `callback_manager` は `CallbackManager` というクラスのインスタンスで、ここに LLM の出力結果を投げる
- おそらくユーザーに出力を見せるなどで使われるよう
- `self.sql_chain(new_inputs, return_only_outputs=True)` で `SQLDatabaseChain` を呼び出しクエリを実行する

結局 SQL を実行するのは `SQLDatabaseChain` であることがわかる。

### SQLDatabaseChain

次に [SQLDatabaseChain](https://github.com/hwchase17/langchain/blob/65c0c73597d93f59c4afb47a398737038f081153/langchain/chains/sql_database/base.py#L67-L103) の中身を見ていこう。

下記は上記リンクの引用である。

```python
def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
    input_text = f"{inputs[self.input_key]} \nSQLQuery:"
    self.callback_manager.on_text(input_text, verbose=self.verbose)
    # If not present, then defaults to None which is all tables.
    table_names_to_use = inputs.get("table_names_to_use")
    table_info = self.database.get_table_info(table_names=table_names_to_use)
    llm_inputs = {
        "input": input_text,
        "top_k": self.top_k,
        "dialect": self.database.dialect,
        "table_info": table_info,
        "stop": ["\nSQLResult:"],
    }
    intermediate_steps = []
    sql_cmd = llm_chain.predict(**llm_inputs)
    intermediate_steps.append(sql_cmd)
    self.callback_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
    result = self.database.run(sql_cmd)
    intermediate_steps.append(result)
    self.callback_manager.on_text("\nSQLResult: ", verbose=self.verbose)
    self.callback_manager.on_text(result, color="yellow", verbose=self.verbose)
    # If return direct, we just set the final result equal to the sql query
    if self.return_direct:
        final_result = result
    else:
        self.callback_manager.on_text("\nAnswer:", verbose=self.verbose)
        input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
        llm_inputs["input"] = input_text
        final_result = llm_chain.predict(**llm_inputs)
        self.callback_manager.on_text(
            final_result, color="green", verbose=self.verbose
        )
    chain_result: Dict[str, Any] = {self.output_key: final_result}
    if self.return_intermediate_steps:
        chain_result["intermediate_steps"] = intermediate_steps
    return chain_result
```

比較的直感的、利用すべき table の情報からクエリを生成し、そのクエリを実行するという流れが記述されている。

### LLMChain

次に LLM を呼び出すところである LLMChain である。
ここで疑問となるのは下記のような input で渡すべき dict は何が決めているのかという点である。

```python
llm_inputs = {
    "input": input_text,
    "top_k": self.top_k,
    "dialect": self.database.dialect,
    "table_info": table_info,
    "stop": ["\nSQLResult:"],
}
sql_cmd = llm_chain.predict(**llm_inputs)
```

`llm_chain` は下記のように必ず prompt とともに初期化される

```python
llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
```

ここで prompt は [prompt.py](https://github.com/hwchase17/langchain/blob/65c0c73597d93f59c4afb47a398737038f081153/langchain/chains/sql_database/prompt.py) で記述されている。

```python
_DEFAULT_TEMPLATE = """_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the tables listed below.

{table_info}

Question: {input}"""
```

ここから、`llm_chain` が要求する dict の値はその template が必要とする値であることがわかる。

## Question Answering with Source

[Question Answering with Source](https://python.langchain.com/en/latest/modules/chains/index_examples/qa_with_sources.html) について説明する。
Question Answering with Source は巨大なデータソースとそのデータソースに対する質問を受け取り、その質問に対する答えを返す。

### 検索対象データの保存

- Chroma に保存
- text はいくつかに chunk していれる

### chain の実行による情報取得

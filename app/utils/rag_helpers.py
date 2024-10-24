from typing import List, Tuple
from sqlalchemy import create_engine, select, Table, MetaData
import os
import dotenv

dotenv.load_dotenv()

engine = create_engine(os.getenv("DB_URL"))
metadata = MetaData()
documents_table = Table('documents', metadata, autoload_with=engine)


class RagHelper:
    def __init__(self):
        self.memory: List[Tuple[str, str]] = []

    def query_model(self, model, question: str):
        # Retrieve documents from the database
        with engine.connect() as connection:
            result = connection.execute(select([documents_table.c.content]))
            documents = [row.content for row in result]

        # Concatenate documents to use as context
        documents_context = "\n".join(documents)

        # Implement Retrieval-Augmented Generation (RAG)
        response = model.answer_with_memory(question, self.memory, documents_context)
        self.memory.append((question, response))
        return response

    def clear_memory(self):
        self.memory = []
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
import dotenv

dotenv.load_dotenv()
engine = create_engine(os.getenv("DB_URL"))
metadata = MetaData()

documents_table = Table(
    'documents', metadata,
    Column('id', Integer, primary_key=True),
    Column('content', String),
)
metadata.create_all(engine)


def read_pdf_file(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(str(len(pages)), "PDF pages loaded.")
    return pages


def ingest_data(data_directory):
    for root, _, files in os.walk(data_directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.pdf'):
                print("Reading file:", file_path)
                pages = read_pdf_file(file_path)
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", "•", " ", ""],
                    chunk_size=1500,
                    chunk_overlap=300,
                    length_function=len
                )
                splits = text_splitter.split_documents(pages)
                index_data(splits)

    print("Data ingestion completed.")


def index_data(splits):
    with engine.connect() as connection:
        for split in splits:
            connection.execute(documents_table.insert().values(content=split.page_content))
    print(f"Total pages indexed: {len(splits)}")


if __name__ == "__main__":
    data_directory = os.path.join(os.getcwd(), "data")
    ingest_data(data_directory)

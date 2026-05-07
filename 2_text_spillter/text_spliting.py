from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('/Users/khusbuagarwal/Desktop/LangChain/9_RAG/1_data_loaders/ml-curriculum.pdf')
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)

print("Total chunks:", len(result))
print("\n--- Chunk 1 ---")
print(result[0].page_content)
print("\n--- Chunk 2 ---")
print(result[1].page_content)
print("\n--- Metadata ---")
print(result[1].metadata)
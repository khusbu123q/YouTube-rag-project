from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# HuggingFace Embeddings instead of OpenAI
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Machine learning is a branch of artificial intelligence that allows computers to learn from data without being explicitly programmed.
It uses algorithms to find patterns in large datasets and make predictions or decisions based on those patterns.
Common machine learning techniques include supervised learning, unsupervised learning, and reinforcement learning.

The stock market is a place where people buy and sell shares of companies.
Investors analyze company performance, economic trends, and market news to make financial decisions.
When a company performs well, its stock price usually goes up, giving investors a profit.

Climate change is one of the biggest challenges facing the world today.
Rising temperatures, melting glaciers, and extreme weather events are becoming more frequent.
Scientists agree that reducing carbon emissions and switching to renewable energy sources are essential steps to protect our planet.

Football is one of the most popular sports in the world, with billions of fans across every continent.
The FIFA World Cup is the biggest football tournament, held every four years and watched by millions of people globally.
Players like Messi and Ronaldo have become global icons, inspiring a new generation of young athletes.
"""

docs = text_splitter.create_documents([sample])

print("Total chunks:", len(docs))
print("\n--- All Chunks ---")
for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}:")
    print(doc.page_content)
    print("-" * 50)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# HuggingFace Model
llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

text = """
Artificial Intelligence is one of the most transformative technologies of the 21st century.
It refers to the simulation of human intelligence in machines that are programmed to think, learn, and problem-solve like humans.
AI is broadly classified into two types - Narrow AI, which is designed to perform a specific task, and General AI, which can perform any intellectual task that a human can do.

Machine Learning is a subset of AI that allows systems to automatically learn and improve from experience without being explicitly programmed.
It focuses on building systems that learn from data, identify patterns, and make decisions with minimal human intervention.
Popular machine learning algorithms include Linear Regression, Decision Trees, Random Forest, and Neural Networks.

Deep Learning is a further subset of Machine Learning that uses neural networks with many layers, called deep neural networks.
These networks are capable of learning representations of data with multiple levels of abstraction.
Deep learning has revolutionized fields like computer vision, natural language processing, and speech recognition.

Natural Language Processing, or NLP, is a branch of AI that deals with the interaction between computers and human language.
It enables machines to read, understand, and generate human language in a meaningful way.
Applications of NLP include chatbots, language translation, sentiment analysis, and text summarization.

The future of AI holds enormous potential. From self-driving cars and personalized medicine to smart cities and automated industries,
AI is set to reshape every aspect of human life. However, it also raises important ethical questions around privacy,
bias, job displacement, and the need for responsible AI development.
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# Perform the split
chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))
print("\n--- All Chunks ---")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
    print("-" * 50)

# Summarize each chunk using HuggingFace
summary_prompt = PromptTemplate(
    template='Write a one line summary of the following text - \n {text}',
    input_variables=['text']
)
summary_chain = summary_prompt | model | parser

print("\n--- CHUNK SUMMARIES ---")
for i, chunk in enumerate(chunks):
    print(f"\nSummary of Chunk {i+1}:")
    print(summary_chain.invoke({'text': chunk}))
    print("-" * 50)
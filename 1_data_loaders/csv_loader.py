import os
from langchain_community.document_loaders import CSVLoader
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

# Load CSV
loader = CSVLoader(file_path='/Users/khusbuagarwal/Desktop/LangChain/9_RAG/1_data_loaders/Social_Network_Ads.csv')
docs = loader.load()

print("Total rows:", len(docs))
print("\n--- Sample Row ---")
print(docs[0].page_content)
print("\n--- Metadata ---")
print(docs[0].metadata)

# Combine all rows into one text
full_text = " ".join([doc.page_content for doc in docs[:20]])  # first 20 rows

# Q&A on CSV
qa_prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following data - \n {text}',
    input_variables=['question', 'text']
)
qa_chain = qa_prompt | model | parser

print("\n--- Q&A 1 ---")
print(qa_chain.invoke({
    'question': 'What is the age range of the users in this dataset?',
    'text': full_text
}))

print("\n--- Q&A 2 ---")
print(qa_chain.invoke({
    'question': 'How many users purchased the product?',
    'text': full_text
}))

# Summary
summary_prompt = PromptTemplate(
    template='Write a short summary of the following dataset - \n {text}',
    input_variables=['text']
)
summary_chain = summary_prompt | model | parser
print("\n--- SUMMARY ---")
print(summary_chain.invoke({'text': full_text}))
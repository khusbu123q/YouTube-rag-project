import os
from langchain_community.document_loaders import PyPDFLoader
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

# Load PDF with full path
loader = PyPDFLoader('/Users/khusbuagarwal/Desktop/LangChain/9_RAG/1_data_loaders/ml-curriculum.pdf')
docs = loader.load()

# Basic Info
print("Total pages:", len(docs))
print("\n--- Page 1 Content ---")
print(docs[0].page_content)
print("\n--- Metadata ---")
print(docs[0].metadata)

# Combine all pages
full_text = " ".join([doc.page_content for doc in docs])

# 1 - SUMMARIZATION
summary_prompt = PromptTemplate(
    template='Write a short and clear summary of the following curriculum - \n {text}',
    input_variables=['text']
)
summary_chain = summary_prompt | model | parser
print("\n--- SUMMARY ---")
print(summary_chain.invoke({'text': full_text}))

# 2 - Q&A
qa_prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)
qa_chain = qa_prompt | model | parser

print("\n--- Q&A 1 ---")
print(qa_chain.invoke({
    'question': 'What topics are covered in Module 3?',
    'text': full_text
}))

print("\n--- Q&A 2 ---")
print(qa_chain.invoke({
    'question': 'What are the ML projects mentioned in the curriculum?',
    'text': full_text
}))

print("\n--- Q&A 3 ---")
print(qa_chain.invoke({
    'question': 'What mathematics topics are covered?',
    'text': full_text
}))

print("\n--- Q&A 4 ---")
print(qa_chain.invoke({
    'question': 'What deep learning frameworks are mentioned?',
    'text': full_text
}))

# 3 - KEY POINTS EXTRACTION
keypoints_prompt = PromptTemplate(
    template='Extract the most important key points from the following curriculum - \n {text}',
    input_variables=['text']
)
keypoints_chain = keypoints_prompt | model | parser
print("\n--- KEY POINTS ---")
print(keypoints_chain.invoke({'text': full_text}))

# 4 - QUIZ GENERATION
quiz_prompt = PromptTemplate(
    template='Generate 5 multiple choice quiz questions from the following curriculum - \n {text}',
    input_variables=['text']
)
quiz_chain = quiz_prompt | model | parser
print("\n--- QUIZ ---")
print(quiz_chain.invoke({'text': full_text}))
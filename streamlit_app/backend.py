import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from deep_translator import GoogleTranslator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

video_store = {}

class VideoRequest(BaseModel):
    video_id: str

class QuestionRequest(BaseModel):
    video_id: str
    question: str

def get_transcript(video_id):
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
        print(f"✅ English transcript found!")
        return transcript
    except Exception:
        try:
            print("English not found. Trying other languages...")
            ytt = YouTubeTranscriptApi()
            available = ytt.list(video_id)
            for t in available:
                print(f"Found: {t.language} ({t.language_code})")
                raw = t.fetch()
                raw_text = " ".join(chunk.text for chunk in raw)
                translator = GoogleTranslator(source='auto', target='en')
                parts = [raw_text[i:i+4500] for i in range(0, len(raw_text), 4500)]
                transcript = " ".join(translator.translate(p) for p in parts)
                print("✅ Translated to English!")
                return transcript
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

def build_pipeline(video_id):
    transcript = get_transcript(video_id)
    if not transcript:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.create_documents([transcript])
    print(f"✅ Total chunks: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = PromptTemplate(
        template="""
        You are a strict assistant analyzing a YouTube video transcript.
        Answer ONLY using exact information from the context below.
        Do NOT add any outside knowledge.
        If the answer is not in the context, say "This topic was not covered in the video."

        Context: {context}
        Question: {question}
        Answer:
        """,
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        | prompt | llm | StrOutputParser()
    )

    print(f"✅ RAG pipeline ready!")
    return chain

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/index")
def index_video(req: VideoRequest):
    if req.video_id in video_store:
        return {"status": "already_indexed", "message": "Video already indexed!"}
    chain = build_pipeline(req.video_id)
    if not chain:
        return {"status": "error", "message": "Could not get transcript."}
    video_store[req.video_id] = chain
    return {"status": "success", "message": "Video indexed successfully!"}

@app.post("/ask")
def ask_question(req: QuestionRequest):
    if req.video_id not in video_store:
        return {"answer": "⏳ Video not indexed yet."}
    chain = video_store[req.video_id]
    answer = chain.invoke(req.question)
    return {"answer": answer}
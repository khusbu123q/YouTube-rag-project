import streamlit as st
import os
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

# ---- Page Config ----
st.set_page_config(
    page_title="YouTube RAG Chat",
    page_icon="🎬",
    layout="centered"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .stTitle { color: #ff0000; }
    .stChatMessage { border-radius: 10px; }
    .status-box {
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ff0000;
        background: #1a1a1a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.title("🎬 YouTube RAG Chat")
st.markdown("Chat with **any YouTube video** in any language using AI!")

# ---- Session State ----
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_video" not in st.session_state:
    st.session_state.current_video = None
if "video_language" not in st.session_state:
    st.session_state.video_language = None

# ---- Helper: Extract Video ID ----
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url.strip()

# ---- Helper: Get Transcript (Any Language) ----
def get_transcript(video_id):
    try:
        # Try English first
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.fetch(video_id, languages=['en'])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        st.session_state.video_language = "English"
        print("✅ English transcript found!")
        return transcript

    except Exception:
        try:
            # Try Hindi specifically
            ytt = YouTubeTranscriptApi()
            transcript_list = ytt.fetch(video_id, languages=['hi'])
            raw_text = " ".join(chunk.text for chunk in transcript_list)
            print("✅ Hindi transcript found! Translating...")

            translator = GoogleTranslator(source='hi', target='en')
            parts = [raw_text[i:i+4500] for i in range(0, len(raw_text), 4500)]
            transcript = " ".join(translator.translate(p) for p in parts)
            st.session_state.video_language = "Hindi → English"
            print("✅ Translated Hindi to English!")
            return transcript

        except Exception:
            try:
                # Try any available language
                print("Trying any available language...")
                ytt = YouTubeTranscriptApi()
                available = ytt.list(video_id)

                for t in available:
                    print(f"Found: {t.language} ({t.language_code})")
                    try:
                        raw = t.fetch()
                        raw_text = " ".join(chunk.text for chunk in raw)

                        if t.language_code == 'en':
                            st.session_state.video_language = "English"
                            return raw_text

                        # Translate to English
                        translator = GoogleTranslator(
                            source=t.language_code,
                            target='en'
                        )
                        parts = [raw_text[i:i+4500] for i in range(0, len(raw_text), 4500)]
                        transcript = " ".join(translator.translate(p) for p in parts)
                        st.session_state.video_language = f"{t.language} → English"
                        print(f"✅ Translated from {t.language} to English!")
                        return transcript

                    except Exception as e:
                        print(f"Failed for {t.language_code}: {e}")
                        continue

            except Exception as e:
                print(f"❌ All attempts failed: {e}")
                return None

# ---- Helper: Build RAG Pipeline ----
def build_pipeline(video_id):
    transcript = get_transcript(video_id)
    if not transcript:
        return None

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.create_documents([transcript])
    print(f"✅ Total chunks: {len(chunks)}")

    # Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # MMR Retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # Prompt
    prompt = PromptTemplate(
        template="""
        You are a strict assistant analyzing a YouTube video transcript.
        Answer ONLY using exact information from the context below.
        Do NOT add any outside knowledge.
        If the answer is not in the context, say "This topic was not covered in the video."

        Context:
        {context}

        Question: {question}

        Answer:
        """,
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG Chain
    chain = (
        RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        | prompt | llm | StrOutputParser()
    )

    print("✅ RAG pipeline ready!")
    return chain

# ---- Sidebar ----
with st.sidebar:
    st.header("🎥 Video Settings")

    video_url = st.text_input(
        "Enter YouTube URL or Video ID",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    if st.button("🚀 Load Video", use_container_width=True):
        if video_url:
            video_id = extract_video_id(video_url)
            with st.spinner("⏳ Fetching transcript and indexing... Please wait..."):
                chain = build_pipeline(video_id)
                if chain:
                    st.session_state.chain = chain
                    st.session_state.current_video = video_id
                    st.session_state.chat_history = []
                    st.success(f"✅ Video indexed successfully!")
                    if st.session_state.video_language:
                        st.info(f"🌐 Language: {st.session_state.video_language}")
                else:
                    st.error("❌ Could not get transcript!")
                    st.markdown("""
                    **Possible reasons:**
                    - Video has no captions
                    - Video is age restricted
                    - Video is private
                    - Try a different video
                    """)
        else:
            st.warning("⚠️ Please enter a YouTube URL first!")

    # Show current video info
    if st.session_state.current_video:
        st.markdown("---")
        st.markdown("**✅ Current Video:**")
        st.code(st.session_state.current_video)
        if st.session_state.video_language:
            st.markdown(f"🌐 **Language:** {st.session_state.video_language}")
        st.video(f"https://www.youtube.com/watch?v={st.session_state.current_video}")

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Load New Video", use_container_width=True):
        st.session_state.chain = None
        st.session_state.chat_history = []
        st.session_state.current_video = None
        st.session_state.video_language = None
        st.rerun()

    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("""
    1. Paste any YouTube URL
    2. Click Load Video
    3. Wait for indexing
    4. Ask any question!
    """)

# ---- Main Chat Area ----
if st.session_state.chain is None:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📥 Step 1")
        st.markdown("Paste a YouTube URL in the sidebar")
    with col2:
        st.markdown("### ⚙️ Step 2")
        st.markdown("Click Load Video and wait for indexing")
    with col3:
        st.markdown("### 💬 Step 3")
        st.markdown("Ask anything about the video!")

    st.info("👈 Enter a YouTube URL in the sidebar to get started!")

else:
    # Show chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    question = st.chat_input("Ask anything about this video...")

    if question:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                answer = st.session_state.chain.invoke(question)
                st.write(answer)

        # Add assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })
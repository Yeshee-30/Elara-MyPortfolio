import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

st.set_page_config(page_title="ELARA — Yeshee Agarwal", layout="wide")

# ---------- LOAD SYSTEM PROMPT ----------
with open("data/systemprompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# ---------- THEME ----------
st.markdown("""
<style>
body { background: radial-gradient(circle at top, #1a0633, #02010a); color:white; }
.hero { text-align:center; padding:60px 20px; }
.hero h1 { font-size:64px; color:#c084fc; text-shadow:0 0 30px #a855f7; }
.hero p { color:#c4b5fd; font-size:18px; }
.glass { background:rgba(15,12,30,.85); border-radius:16px; padding:25px; box-shadow:0 0 40px rgba(168,85,247,.2); }
.card { background:rgba(20,15,40,.9); border-radius:14px; padding:20px; margin:10px; box-shadow:0 0 25px rgba(168,85,247,.2); }
.card h3 { color:#c084fc; }
.footer { text-align:center; color:#a78bfa; margin-top:50px; }
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
<h1>ELARA</h1>
<p>Yeshee Agarwal’s Digital Twin</p>
<p>Ask anything about her AI, ML, projects, and experience.</p>
</div>
""", unsafe_allow_html=True)

# ---------- LOAD DOCUMENTS ----------
@st.cache_resource
def load_vectorstore():
    docs = []
    for file in os.listdir("data"):
        docs.extend(TextLoader(f"data/{file}", encoding="utf-8").load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ---------- LLM ----------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.4
)

# ---------- CHAT ----------
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("## 💬 Talk to ELARA")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "elara_input" not in st.session_state:
    st.session_state.elara_input = ""

user_input = st.text_input("Ask ELARA about Yeshee…", key="elara_input")
send = st.button("Send ✨")

if send and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    # retrieve relevant docs
    docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([d.page_content for d in docs])

    # build chat history
    history = ""
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            history += f"User: {m['content']}\n"
        else:
            history += f"ELARA: {m['content']}\n"

    # final prompt
    prompt = f"""
{SYSTEM_PROMPT}

Conversation so far:
{history}

Context:
{context}

User question:
{user_input}

Answer as ELARA:
"""

    answer = llm.invoke(prompt).content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ---------- PROJECTS ----------
st.markdown("## 🚀 Projects")
cols = st.columns(3)
cols[0].markdown("<div class='card'><h3>🚗 Pothole Detection</h3>CNN based assistive driving system (95%).</div>", unsafe_allow_html=True)
cols[1].markdown("<div class='card'><h3>📄 DocuSnap</h3>Enterprise GenAI document intelligence.</div>", unsafe_allow_html=True)
cols[2].markdown("<div class='card'><h3>🧠 Answer Sheet AI</h3>Agentic AI for evaluation.</div>", unsafe_allow_html=True)
st.markdown("<div class='card'><h3>👁 Cataract Detection</h3>Medical CNN detection system.</div>", unsafe_allow_html=True)

# ---------- EXPERIENCE ----------
st.markdown("## 💼 Experience")
st.markdown("""
<div class='card'><h3>Silicon Interfaces</h3>Built ANN models for chip fault simulation with 90%+ accuracy.</div>
<div class='card'><h3>L&T Technology Services</h3>Built GenAI, RAG, DocuSnap and enterprise AI systems showcased to Nordic leadership.</div>
""", unsafe_allow_html=True)

# ---------- CONTACT ----------
st.markdown("## 📬 Contact")
st.markdown("""
<div class='card'>
<a href="https://github.com/yeshee-30" target="_blank">GitHub</a><br>
<a href="https://www.linkedin.com/in/yeshee-agarwala-abba96280/" target="_blank">LinkedIn</a><br>
<a href="mailto:agarwalyeshee364@gmail.com">Email</a>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='footer'>Built with 💜 by Yeshee Agarwal</div>", unsafe_allow_html=True)

import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()


# Silence LangChain junk
os.environ["LANGCHAIN_TRACING_V2"] = "false"


st.set_page_config(page_title="ELARA — Yeshee Agarwal", layout="wide")


# ----------------- LOAD SYSTEM PROMPT -----------------
with open("data/systemprompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


# ----------------- PURPLE THEME -----------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #1a0633, #02010a);
    color: white;
}
.hero {
    text-align: center;
    padding: 60px 20px 40px 20px;
}
.hero h1 {
    font-size: 64px;
    color: #c084fc;
    text-shadow: 0 0 30px #a855f7;
}
.hero p {
    font-size: 18px;
    color: #c4b5fd;
}
.glass {
    background: rgba(15, 12, 30, 0.8);
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 0 40px rgba(168,85,247,0.2);
}
.card {
    background: rgba(20,15,40,0.9);
    border-radius: 14px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 0 25px rgba(168,85,247,0.2);
}
.card h3 {
    color: #c084fc;
}
.footer {
    text-align:center;
    color:#a78bfa;
    margin-top:50px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <h1>ELARA</h1>
    <p>Yeshee Agarwal's Digital Twin</p>
    <p>Ask anything about her AI, ML, projects, and experience.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
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


# ---------------- LLM ----------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.4
)


# ---------------- CHAT ----------------
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
    st.session_state.messages.append({"role":"user","content":user_input})


    # Manual conversational retrieval - NO CHAINS
    full_prompt = SYSTEM_PROMPT + "\n\nUser question: " + user_input
    
    # Get chat history for context
    chat_history = []
    for msg in st.session_state.messages[:-1]:  # Exclude current user input
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    
    # Contextualize question using chat history
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, rephrase the latest user question to be a standalone question that incorporates relevant context from history."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", full_prompt),
    ])
    
    contextualized_question = context_prompt | llm | StrOutputParser()
    standalone_question = contextualized_question.invoke({"chat_history": chat_history})
    
    # Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(standalone_question)
    
    # Format context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Generate final response
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + f"\n\nRelevant context:\n{context}\n\nAnswer the question based only on the context above. If you don't know the answer, say so."),
        ("human", user_input),
    ])
    
    response_chain = qa_prompt | llm | StrOutputParser()
    response = response_chain.invoke({})
    
    st.session_state.messages.append({"role":"assistant","content":response})
    st.rerun()


st.markdown("</div>", unsafe_allow_html=True)


# ---------------- PROJECTS ----------------
st.markdown("## 🚀 Projects")
cols = st.columns(3)


with cols[0]:
    st.markdown("<div class='card'><h3>🚗 Pothole Detection</h3>CNN-based assistive driving system with 94–95% accuracy.</div>", unsafe_allow_html=True)
with cols[1]:
    st.markdown("<div class='card'><h3>📄 DocuSnap</h3>Enterprise GenAI for extracting structured data from documents and images.</div>", unsafe_allow_html=True)
with cols[2]:
    st.markdown("<div class='card'><h3>🧠 Answer Sheet AI</h3>Agentic AI that evaluates descriptive answers and reduces teacher workload.</div>", unsafe_allow_html=True)


st.markdown("<div class='card'><h3>👁 Cataract Detection</h3>Medical CNN system for eye disease detection.</div>", unsafe_allow_html=True)


# ---------------- EXPERIENCE ----------------
st.markdown("## 💼 Experience")
st.markdown("""
<div class='card'><h3>Silicon Interfaces</h3>Built and trained a feed-forward deep learning model for predicting stuck-at faults in semiconductor circuits using fault-injected test vectors.  
Worked extensively with neural network architectures, activation functions (ReLU, Sigmoid), and optimizers such as RMSProp and Adam to stabilize and improve model performance.  
Performed hyperparameter tuning on learning rate, epochs, and loss functions to achieve high generalization accuracy.  
The final model achieved over 90% accuracy, significantly improving the reliability and efficiency of chip fault simulation.</div>
<div class='card'><h3>L&T Technology Services</h3>Designed and implemented multiple enterprise-grade Generative AI and Agentic AI solutions for automating data extraction, document understanding, and decision-making workflows.  
Built a Smart AI system capable of processing emails, PDFs, scanned documents, handwritten labels, and images, converting unstructured data into structured, machine-readable outputs using LLMs, RAG pipelines, and FastAPI.  
Developed a foundational GenAI architecture that became the base layer for multiple internal solutions, including DocuSnap and automated document intelligence pipelines.  
The system was showcased to the Nordic Region leadership and adopted as the foundation for an internal L&T product, positioning it as an enterprise-scale AI capability within the organization.</div>
""", unsafe_allow_html=True)


# ---------------- CONTACT ----------------
st.markdown("## 📬 Contact")


st.markdown("""
<div class='card'>
<b>GitHub:</b> 
<a href="https://github.com/yeshee-30" target="_blank" style="color:#c084fc; text-decoration:none;">
github.com/yeshee-30
</a>
<br><br>


<b>LinkedIn:</b> 
<a href="https://www.linkedin.com/in/yeshee-agarwala-abba96280/" target="_blank" style="color:#c084fc; text-decoration:none;">
linkedin.com/in/yeshee-agarwala
</a>
<br><br>


<b>Email:</b> 
<a href="mailto:agarwalyeshee364@gmail.com" style="color:#c084fc;">
[agarwalyeshee364@gmail.com](mailto:agarwalyeshee364@gmail.com)
</a>
</div>
""", unsafe_allow_html=True)


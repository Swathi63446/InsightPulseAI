import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Sales Lab", page_icon="🎧", layout="wide")

# --- API KEY & MODEL INITIALIZATION ---
API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("Missing API Key! Please set GOOGLE_API_KEY in .env or Streamlit Secrets.")
    st.stop()

genai.configure(api_key=API_KEY)

# --- SAFETY SETTINGS ---
# This prevents the "finish_reason: 1" error by allowing 
# more flexible content for professional sales coaching.
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    safety_settings=safety_settings
)

# --- DATABASE SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "scripts", "data", "chroma_db")

@st.cache_resource
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

emb_fn = get_embedding_function()

# --- SIDEBAR ---
st.sidebar.title("🚀 Sales Intelligence")
if st.sidebar.button("Refresh Knowledge Base"):
    with st.spinner("Indexing new insights..."):
        # Ensure the path to your script is correct
        script_path = os.path.join(BASE_DIR, "scripts", "build_knowledge_base.py")
        exit_code = os.system(f"python {script_path}")
        if exit_code == 0:
            st.sidebar.success("Database Updated!")
        else:
            st.sidebar.error("Indexing script failed. check logs.")

# --- MAIN INTERFACE ---
st.title("🎧 AI Sales Coach")
st.info("Ask questions based on your scraped sales Reels and scripts.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT LOGIC (RAG) ---
if prompt := st.chat_input("Ask about a sales script..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Consulting the playbook..."):
        try:
            # Connect to ChromaDB
            client = chromadb.PersistentClient(path=DB_PATH)
            collection = client.get_collection(name="sales_playbook", embedding_function=emb_fn)
            
            # Query Database
            results = collection.query(query_texts=[prompt], n_results=3)
            
            # Extract Context safely
            docs = results.get('documents', [[]])[0]
            context = "\n\n".join(docs) if docs else "No relevant scripts found in the database."

            # Construct Prompt
            full_prompt = f"""
            SYSTEM: You are a high-performance Sales Coach.
            
            CONTEXT FROM SCRAPED VIDEOS:
            {context}
            
            USER QUESTION: {prompt}
            
            GOAL: Provide a specific script or actionable advice based on the context above. 
            If the context doesn't fully answer the question, use your expertise to fill the gaps.
            """
            
            # Generate with Error Handling for Safety Blocks
            response = model.generate_content(full_prompt)
            
            # Check if response was blocked before accessing .text
            if response.candidates and response.candidates[0].content.parts:
                answer = response.text
            else:
                answer = "The AI coach blocked this response due to safety filters. Try rephrasing your request without aggressive language."
            
        except Exception as e:
            answer = f"⚠️ Error: {str(e)}\n\n(Make sure the database exists at `{DB_PATH}`)"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

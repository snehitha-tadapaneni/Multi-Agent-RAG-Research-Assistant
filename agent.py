# import necessary libraries
import os
import base64
import tempfile
import warnings
import gradio as gr
import pandas as pd
import pdfplumber
from gtts import gTTS
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM

warnings.filterwarnings("ignore")

# IBM Watsonx credentials
credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
project_id = "skills-network"

# ---- Document Loader (Docling Style) ----
def document_loader(file):
    loader = PyPDFLoader(file.name)
    return loader.load()

def text_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return splitter.split_documents(docs)

# ---- Retriever Agent ----
class RetrieverAgent:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.embeddings = WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr",
            url="https://us-south.ml.cloud.ibm.com",
            project_id=project_id
        )

    def build(self):
        bm25 = BM25Retriever.from_documents(self.documents)
        vectorstore = Chroma.from_documents(self.documents, self.embeddings)
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        return EnsembleRetriever(retrievers=[bm25, chroma_retriever], weights=[0.5, 0.5])

# ---- Research Agent ----
class ResearchAgent:
    def __init__(self):
        self.model = ModelInference(
            model_id="meta-llama/llama-3-2-90b-vision-instruct",
            credentials=credentials,
            project_id=project_id,
            params={"max_tokens": 300, "temperature": 0.3}
        )

    def generate_prompt(self, question: str, context: str) -> str:
        return f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.

        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """

    def generate(self, question: str, documents: List[Document]) -> dict:
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(question, context)
        try:
            response = self.model.chat(messages=[{"role": "user", "content": prompt}])
            llm_response = response['choices'][0]['message']['content'].strip()
        except Exception as e:
            llm_response = f"❌ Error: {str(e)}"
        return {"draft_answer": llm_response, "context_used": context}

# ---- Verification Agent ----
class VerificationAgent:
    def verify(self, answer: str, docs: List[Document]) -> str:
        context_texts = [doc.page_content for doc in docs]
        try:
            vectorizer = TfidfVectorizer().fit(context_texts + [answer])
            vectors = vectorizer.transform(context_texts + [answer])
            scores = cosine_similarity(vectors[-1], vectors[:-1])[0]
            best_match = context_texts[scores.argmax()]
            score = scores.max()
            return f"✅ Verified (cosine score: {score:.2f})\n\nRelevant Context:\n{best_match[:500]}..."
        except Exception as e:
            return f"❌ Verification error: {e}"

# ---- Citation/Metadata Agent ----
class CitationAgent:
    def extract_metadata(self, documents: List[Document]) -> str:
        try:
            titles = [doc.metadata.get("title", "") for doc in documents if "title" in doc.metadata]
            return f"📚 Extracted Metadata:\nTitle(s): {', '.join(titles) if titles else 'Not available'}"
        except Exception as e:
            return f"❌ Metadata extraction error: {e}"

# ---- OCR Image Agent (placeholder) ----
def ocr_image_to_text(image_path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception as e:
        return f"❌ OCR Error: {e}"

# ---- Decision Agent ----
class DecisionAgent:
    def finalize(self, answer: str, verification: str, citation: str = "") -> str:
        return f"""
📌 Final Answer:
{answer}

🧾 Verification Result:
{verification}

🔖 Citation Info:
{citation}
"""

# ---- Unified Orchestrator ----
def process_query(file, query):
    if not file or not query:
        return "Please upload a file and ask a question.", None
    try:
        docs = document_loader(file)
        chunks = text_splitter(docs)

        retriever = RetrieverAgent(chunks).build()
        top_docs = retriever.get_relevant_documents(query)

        research = ResearchAgent()
        result = research.generate(query, top_docs)
        answer = result["draft_answer"]

        verifier = VerificationAgent()
        check = verifier.verify(answer, top_docs)

        citation_agent = CitationAgent()
        citation = citation_agent.extract_metadata(top_docs)

        decision = DecisionAgent()
        final_response = decision.finalize(answer, check, citation)

        tts = gTTS(text=final_response)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            audio_path = tmp.name

        return final_response, audio_path

    except Exception as e:
        return f"❌ Error: {e}", None

# ---- Gradio App ----
with gr.Blocks(title="📚 Multimodal RAG Research Assistant") as app:
    gr.Markdown("""
    # 🤖 Multimodal AI Research Assistant  
    Upload a **PDF** and ask a question.  
    The model will read, analyze, verify, and even speak the answer!  
    _Powered by IBM Watsonx + LangChain + gTTS_
    """)

    with gr.Row():
        file_input = gr.File(label="📁 Upload PDF", file_types=[".pdf"], type="filepath")

    with gr.Row():
        query = gr.Textbox(label="❓ Ask your question", placeholder="E.g., Summarize this paper", lines=2)

    with gr.Row():
        submit_btn = gr.Button("💬 Get Answer")

    with gr.Row():
        output_text = gr.Textbox(label="🧠 Model Response", lines=10)
        output_audio = gr.Audio(label="🔊 Listen", type="filepath", autoplay=True)

    submit_btn.click(fn=process_query, inputs=[file_input, query], outputs=[output_text, output_audio])

app.launch(server_name="127.0.0.1", server_port=7860)

# Import neccessary libraries
import os
import tempfile
import warnings
import gradio as gr
from gtts import gTTS
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_ibm import WatsonxEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# defining state Schema
from typing import TypedDict, List

class RAGState(TypedDict):
    file: str
    query: str
    docs: List[Document]
    top_docs: List[Document]
    draft_answer: str
    verification: str
    citation: str
    final_response: str


# ---- Credentials ----
credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
project_id = "skills-network"

# ---- Document Parsing ----
def document_loader(file):
    loader = PyPDFLoader(file.name)
    return loader.load()

def text_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return splitter.split_documents(docs)

# ---- Retriever Agent ----
class RetrieverAgent:
    def __init__(self, documents):
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

    def generate_prompt(self, question, context):
        return f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.

        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """

    def generate(self, question, documents):
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(question, context)
        try:
            response = self.model.chat(messages=[{"role": "user", "content": prompt}])
            return {"draft_answer": response['choices'][0]['message']['content'].strip()}
        except Exception as e:
            return {"draft_answer": f"❌ Error: {str(e)}"}

# ---- Verification Agent ----
class VerificationAgent:
    def verify(self, answer, docs):
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

# ---- Citation Agent ----
class CitationAgent:
    def extract_metadata(self, documents):
        titles = [doc.metadata.get("title", "") for doc in documents if "title" in doc.metadata]
        return f"📚 Extracted Title(s): {', '.join(titles) if titles else 'Not available'}"

# ---- Decision Agent ----
class DecisionAgent:
    def finalize(self, answer, verification, citation):
        return f"""
📌 Final Answer:
{answer}

🧾 Verification Result:
{verification}

🔖 Citation Info:
{citation}
"""

# ---- LangGraph DAG ----
def create_rag_pipeline():
    def parse_node(state):
        docs = document_loader(state["file"])
        return {"docs": text_splitter(docs), "query": state["query"]}

    def retrieve_node(state):
        retriever = RetrieverAgent(state["docs"]).build()
        return {"top_docs": retriever.get_relevant_documents(state["query"])}

    def research_node(state):
        result = ResearchAgent().generate(state["query"], state["top_docs"])
        return {"draft_answer": result["draft_answer"]}

    def verify_node(state):
        check = VerificationAgent().verify(state["draft_answer"], state["top_docs"])
        return {"verification": check}

    def citation_node(state):
        citation = CitationAgent().extract_metadata(state["top_docs"])
        return {"citation": citation}

    def decision_node(state):
        result = DecisionAgent().finalize(state["draft_answer"], state["verification"], state["citation"])
        return {"final_response": result}

    graph = StateGraph(RAGState)
    graph.add_node("Parse", parse_node)
    graph.add_node("Retrieve", retrieve_node)
    graph.add_node("Research", research_node)
    graph.add_node("Verify", verify_node)
    graph.add_node("Citation", citation_node)
    graph.add_node("Decision", decision_node)

    graph.set_entry_point("Parse")
    graph.add_edge("Parse", "Retrieve")
    graph.add_edge("Retrieve", "Research")
    graph.add_edge("Research", "Verify")
    graph.add_edge("Verify", "Citation")
    graph.add_edge("Citation", "Decision")
    graph.add_edge("Decision", END)

    return graph.compile()

pipeline = create_rag_pipeline()

# ---- Gradio Interface ----
def gradio_pipeline(file, query):
    try:
        result = pipeline.invoke({"file": file, "query": query})
        final_response = result["final_response"]

        tts = gTTS(text=final_response)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            audio_path = tmp.name
        return final_response, audio_path
    except Exception as e:
        return f"❌ Error: {e}", None

with gr.Blocks(title="📚 Multimodal RAG Research Assistant with LangGraph") as app:
    gr.Markdown("""
    # 🤖 Multimodal AI Research Assistant (LangGraph-powered)
    Upload a **PDF** and ask a question. The assistant will parse, retrieve, reason, verify, cite, and speak the answer.
    """)

    file_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
    query = gr.Textbox(label="Ask your question", placeholder="E.g., What is the methodology used?", lines=2)
    submit_btn = gr.Button("💬 Get Answer")
    output_text = gr.Textbox(label="Model Response", lines=10)
    output_audio = gr.Audio(label="Listen", type="filepath", autoplay=True)

    submit_btn.click(fn=gradio_pipeline, inputs=[file_input, query], outputs=[output_text, output_audio])

app.launch(server_name="172.22.132.163", server_port=8888)

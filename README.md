# Multi-Agent-RAG-Research-Assistant
I have built a Multimodal Research Assistant using Watson by IBM, LangChain and Gradio. This assistant can read research papers, retrieve relevant information, generate answers using LLMs, verify them, and even speak them aloud all from a single interface!

# Why Multi-Agent RAG?
Why Multi-Agent RAG?
Traditional RAG pipelines typically involve:
 1. Retrieving relevant documents
 2. Passing them to a language model
 3. Getting a final response

But real research involves more relevance checking, factual verification, and decision-making.

So, I broke this process down into a multi-agent architecture, where each agent handles one specialized task.
Here's how the agents works:

1. Document parser Agent: Parse both structured and unstructured PDF's. Can use Docling here (by IBM)       . This makes the assistant resilient from extractng content even from scanned or OCR based documents.
2. Retriever Agent: Pull the most relevant chuncks from the PDF using a hybrid strategy. I combined BM25 and ChromaDB for a robust recall of relevant content.
3. Research Agent: Generate a clear, structured, factual answer grounded in the retrieved content. (Any LLm can be used based on the capability. Here, i used IBM Watsonx vision instruct).
4. Verfiation Agent: Double check the LLM's response against the context.
5. Decision Agent: Combine the draft answer and verification to create a final human readable response.
6. Test-to-Speech Agent: Reads the answer aloud for acccessibility and convenience.

 


**Installation and Requirements:**
```
pip install -r requirments.txt
pip install gradio langchain ibm-watsonx-ai gtts chromadb pytesseract pdfplumber
```

**How to Run?**
```
python app.py
```

This assistant reflects how modular AI agents can collaborate to enhance research workflows. By combining retrieval, generation, verification, and speech, we create a richer, more accurate and accessible AI experience.


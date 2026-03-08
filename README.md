# 🇵🇰 Pakistan Studies RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot designed to answer questions about Pakistan Studies and History.

The chatbot retrieves information from a Pakistan Studies textbook and generates answers using an LLM. If the retrieved context is insufficient, the system automatically performs a web search using Tavily to gather additional information.

This hybrid approach helps produce accurate, contextual, and reliable responses while minimizing hallucinations.

## 🚀 Features

### 📚 RAG-Based Question Answering
Retrieves relevant passages from a Pakistan Studies history book.

### 🌐 Automatic Web Search Fallback
If the local context is insufficient, the system performs a Tavily web search.

### 🧠 Context-Aware Responses
Answers are generated using retrieved information to reduce hallucinations.

### 📄 OCR for Scanned PDFs
Uses Tesseract OCR to extract text from scanned textbook pages.

### 🔎 Semantic Retrieval
Uses embeddings and a vector database to find the most relevant sections of the textbook.

## 🏗 System Architecture

```mermaid
flowchart TD
    A[User Question] --> B[Retrieve Context from Pakistan Studies Book]
    B --> C{Is Context Enough?}
    C -->|Yes| D[Generate Answer using LLM]
    C -->|No| E[Perform Tavily Web Search]
    E --> F[Add Web Context to Prompt]
    F --> D
    D --> G[Final Response to User]

## 🛠 Tech Stack

### Python  

### LangChain / RAG Pipeline

### Vector Database (Pinecone)

### Tesseract OCR

### Tavily Search API

### Large Language Models (Groq)

# 📚 Notes Assistant (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions about their contents.

The application processes documents using semantic search, retrieves the most relevant context, and generates accurate answers using Google's Gemini model.

---

## ✨ Features

- 📄 Upload and index PDF documents
- ✂️ Automatic document chunking
- 🧠 Semantic embeddings using Sentence Transformers
- 🗂️ Vector search powered by ChromaDB
- 🤖 AI-powered answers using Google Gemini
- 📚 Source attribution for every answer
- 🔄 Replace previously indexed document
- 🧪 Comprehensive unit test suite
- 🎨 Clean Streamlit interface

---

## 🖥️ Demo

<p align="center">
  <video src="https://github.com/nilaysarma/notes-assistant/raw/refs/heads/main/demo.mp4" width="450" controls></video>
</p>

### Upload a document

Upload any PDF and index it into the vector database.

### Ask questions

Ask questions in natural language.

Example:

> What is Gradient Descent?

### View Sources

Every answer includes the document chunks used to generate the response.

---

## 🏗️ Architecture

```text
                    Streamlit UI
                          │
                          ▼
                  RAG Service Layer
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
    Retrieval                         LLM Generation
        │                                   │
        ▼                                   ▼
    ChromaDB                          Gemini API
        ▲
        │
Sentence Transformer Embeddings
        ▲
        │
PDF Loader & Text Splitter
```

The project follows a layered architecture where the frontend only communicates with the service layer. Business logic is completely separated from the user interface.

---

## 🛠️ Tech Stack

### Frontend

- Streamlit

### Backend

- Python 3.12
- Pydantic
- LangChain Text Splitters

### AI / RAG

- Sentence Transformers
- ChromaDB
- Google Gemini

### Testing

- Pytest

---

## 📁 Project Structure

```text
app/
│
├── core/
│   └── config.py
│
├── llm/
│   └── gemini.py
│
├── models/
│   ├── answer_result.py
│   ├── chunk.py
│   ├── indexing_result.py
│   └── page.py
│
├── rag/
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── prompt_builder.py
│
├── services/
│   └── rag_service.py
│
└── utils/
    └── logger.py

tests/

data/

streamlit_app.py
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/nilaysarma/notes-assistant.git

cd notes-assistant
```

Install dependencies

```bash
uv sync
```

Create a `.env` file

```env
GOOGLE_API_KEY=your_api_key_here
```

Run the application

```bash
uv run streamlit run streamlit_app.py
```

or

```bash
uv run streamlit run streamlit_app.py --server.fileWatcherType none
```

---

## 🧪 Running Tests

Run the complete test suite

```bash
uv run python -m pytest
```

---

## 📖 How it Works

1. Upload a PDF document.
2. Extract text from each page.
3. Split text into overlapping chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in ChromaDB.
6. Embed the user's question.
7. Retrieve the most relevant chunks.
8. Build a prompt using the retrieved context.
9. Generate the final answer using Gemini.
10. Display the answer together with its supporting sources.

---

## 🚀 Future Improvements

- Chrome Extension for indexing web pages
- Multi-document support
- Embedding progress bar

---

## 📄 License

This project is licensed under the MIT License.

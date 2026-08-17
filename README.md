#🎓 School AI Copilot

> Learn from your own study materials.

School AI Copilot is a simple RAG-based educational assistant that allows students to upload their study notes and interact with them using AI.

Instead of asking an AI system to answer from general knowledge, the application retrieves relevant information from the student's uploaded documents and uses that context to generate responses.

## 🚨 The Problem

Students often have their learning material spread across PDFs, lecture notes, and revision documents.

When studying, they may need to:

- Find answers inside long notes
- Create practice questions
- Summarize chapters
- Make flashcards
- Plan their revision

School AI Copilot brings these tasks into one simple application.

## 💡 The Solution

Students upload a PDF containing their study material.

The application:

1. Loads the PDF.
2. Splits the document into smaller chunks.
3. Converts the chunks into embeddings.
4. Stores them in a vector database.
5. Retrieves relevant sections when the student asks something.
6. Uses an LLM to generate a response based on the retrieved context.

## ✨ Features

### 💬 Ask Question
Ask questions about the uploaded study material.

### 📝 Generate Quiz
Generate 10 multiple-choice questions from the relevant material.

### 📖 Summarize
Create a short summary of a topic or section.

### 🗂️ Flashcards
Generate question-and-answer flashcards for revision.

### 📅 Study Plan
Generate a 7-day study plan based on the uploaded material.

## 🧠 How It Works

```text
Student PDF
    ↓
PyPDFLoader
    ↓
Text Splitting
    ↓
Hugging Face Embeddings
    ↓
Chroma Vector Database
    ↓
Retriever
    ↓
Relevant Context
    ↓
Groq LLM
    ↓
Study Response

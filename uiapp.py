from dotenv import load_dotenv
import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# SETUP
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="School AI Copilot",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 School AI Copilot")

st.write("Upload your study notes and ask questions or generate study materials.")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📄 Upload Notes")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)


# ============================================================
# MAIN APP
# ============================================================

if uploaded_file is None:

    st.info("👈 Upload a PDF from the sidebar to get started.")

else:

    # --------------------------------------------------------
    # SAVE PDF
    # --------------------------------------------------------

    os.makedirs("data", exist_ok=True)

    pdf_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    # --------------------------------------------------------
    # LOAD PDF
    # --------------------------------------------------------

    with st.spinner("Processing your PDF..."):

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()


        # ----------------------------------------------------
        # SPLIT DOCUMENT
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)


        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )


        # ----------------------------------------------------
        # VECTOR DATABASE
        # ----------------------------------------------------

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="vector_db"
        )


        # ----------------------------------------------------
        # RETRIEVER
        # ----------------------------------------------------

        retriever = vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )


        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        llm = ChatGroq(
            model="openai/gpt-oss-120b"
        )


    st.success(f"✅ {uploaded_file.name} is ready!")


    # ========================================================
    # PROMPTS
    # ========================================================

    prompt_template = ChatPromptTemplate.from_template(
        """
You are a helpful school assistant.

Use ONLY the provided context.

If the answer is not found in the context,
reply:

"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}
"""
    )


    quiz_prompt = ChatPromptTemplate.from_template(
        """
You are a school teacher.

Use ONLY the context below.

Context:
{context}

Generate 10 multiple-choice questions.

Each question should include:

- Four options
- Correct answer
- Short explanation
"""
    )


    study_prompt = ChatPromptTemplate.from_template(
        """
Create a 7-day study plan.

Context:

{context}

Include:

Topics

Revision

Practice Questions
"""
    )


    summary_prompt = ChatPromptTemplate.from_template(
        """
Summarize the following chapter.

Context:

{context}

Keep the summary under 200 words.
"""
    )


    flashcard_prompt = ChatPromptTemplate.from_template(
        """
Create flashcards.

Context:

{context}

Format:

Question:

Answer:
"""
    )


    # ========================================================
    # CHAINS
    # ========================================================

    qa_chain = prompt_template | llm
    quiz_chain = quiz_prompt | llm
    study_chain = study_prompt | llm
    summary_chain = summary_prompt | llm
    flashcard_chain = flashcard_prompt | llm


    # ========================================================
    # MODE
    # ========================================================

    st.sidebar.divider()

    st.sidebar.header("🧠 Choose Mode")

    mode = st.sidebar.radio(
        "Select one:",
        [
            "Ask Question",
            "Generate Quiz",
            "Summarize",
            "Flashcards",
            "Study Plan"
        ]
    )


    # ========================================================
    # QUERY
    # ========================================================

    query = st.text_input(
        "Enter your query",
        placeholder="Example: Explain gravitational potential energy."
    )


    # ========================================================
    # BUTTON
    # ========================================================

    if st.button("Generate", type="primary"):

        if query.strip() == "":

            st.warning("Please enter a query.")

        else:

            with st.spinner("Thinking..."):

                # Retrieve relevant documents
                documents = retriever.invoke(query)

                # Combine retrieved text
                context = "\n\n".join(
                    doc.page_content
                    for doc in documents
                )


                # ------------------------------------------------
                # ASK QUESTION
                # ------------------------------------------------

                if mode == "Ask Question":

                    response = qa_chain.invoke(
                        {
                            "context": context,
                            "question": query
                        }
                    )


                # ------------------------------------------------
                # QUIZ
                # ------------------------------------------------

                elif mode == "Generate Quiz":

                    response = quiz_chain.invoke(
                        {
                            "context": context
                        }
                    )


                # ------------------------------------------------
                # SUMMARY
                # ------------------------------------------------

                elif mode == "Summarize":

                    response = summary_chain.invoke(
                        {
                            "context": context
                        }
                    )


                # ------------------------------------------------
                # FLASHCARDS
                # ------------------------------------------------

                elif mode == "Flashcards":

                    response = flashcard_chain.invoke(
                        {
                            "context": context
                        }
                    )


                # ------------------------------------------------
                # STUDY PLAN
                # ------------------------------------------------

                elif mode == "Study Plan":

                    response = study_chain.invoke(
                        {
                            "context": context
                        }
                    )


            # ====================================================
            # DISPLAY RESPONSE
            # ====================================================

            st.divider()

            st.subheader("🤖 School AI Copilot")

            st.write(response.content)
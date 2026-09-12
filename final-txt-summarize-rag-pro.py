import os
import pickle
import faiss
import streamlit as st

from sentence_transformers import SentenceTransformer

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Sumifyer",
    page_icon="📈"
)

st.title("Sumifyer: News Research Tool 📈")


# ============================================================
# SIDEBAR - URL INPUT
# ============================================================

st.sidebar.title("News Article URLs")

urls = []

for i in range(3):
    url = st.sidebar.text_input(
        f"URL {i + 1}"
    )
    urls.append(url)

process_url_clicked = st.sidebar.button(
    "Process URLs"
)


# ============================================================
# FILE
# ============================================================

file_path = "faiss_store.pkl"

main_placeholder = st.empty()


# ============================================================
# SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_encoder():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


encoder = load_encoder()


# ============================================================
# GEMINI
# ============================================================

@st.cache_resource
def load_llm():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        api_key = st.secrets.get("GOOGLE_API_KEY")

    if not api_key:
        st.error("GOOGLE_API_KEY is not configured.")
        st.stop()

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=api_key,
        max_tokens=500
    )


llm = load_llm()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are a helpful news research assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say that you don't know.

Context:

{context}
"""
    ),

    (
        "human",
        "{question}"
    )

])


chain = prompt | llm


# ============================================================
# PROCESS URLS
# ============================================================

if process_url_clicked:

    urls = [
        url
        for url in urls
        if url.strip()
    ]

    if not urls:

        st.error(
            "Please enter at least one URL."
        )

        st.stop()


    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    main_placeholder.text(
        "1️⃣ Loading data..."
    )

    loader = UnstructuredURLLoader(
        urls=urls
    )

    data = loader.load()

    st.success(
        f"Documents loaded: {len(data)}"
    )


    # ========================================================
    # 2. TEXT SPLITTING
    # ========================================================

    main_placeholder.text(
        "2️⃣ Splitting text into chunks..."
    )

    text_splitter = RecursiveCharacterTextSplitter(

        separators=[
            "\n\n",
            "\n",
            ".",
            ",",
            " "
        ],

        chunk_size=200,

        chunk_overlap=0
    )


    chunks = text_splitter.split_documents(
        data
    )


    if not chunks:

        st.error(
            "No text chunks were created from the URLs."
        )

        st.stop()


    st.success(
        f"Chunks created: {len(chunks)}"
    )


    # ========================================================
    # SHOW FIRST CHUNK
    # ========================================================

    with st.expander(
        "View first chunk"
    ):

        st.write(
            chunks[0].page_content
        )

        st.write(
            "Characters:",
            len(chunks[0].page_content)
        )


    # ========================================================
    # 3. CREATE EMBEDDINGS
    # ========================================================

    main_placeholder.text(
        "3️⃣ Creating embeddings..."
    )


    texts = [
        chunk.page_content
        for chunk in chunks
    ]


    vectors = encoder.encode(
        texts,
        convert_to_numpy=True
    )


    st.write(
        "Embedding shape:",
        vectors.shape
    )


    # ========================================================
    # 4. CREATE FAISS INDEX
    # ========================================================

    main_placeholder.text(
        "4️⃣ Building FAISS index..."
    )


    dimension = vectors.shape[1]


    index = faiss.IndexFlatL2(
        dimension
    )


    index.add(
        vectors
    )


    st.success(
        f"FAISS vectors stored: {index.ntotal}"
    )


    # ========================================================
    # 5. SAVE FAISS + DOCUMENTS
    # ========================================================

    with open(
        file_path,
        "wb"
    ) as f:

        pickle.dump(
            {
                "index": index,
                "docs": chunks
            },
            f
        )


    main_placeholder.success(
        "Processing complete! ✅"
    )


# ============================================================
# QUESTION
# ============================================================

st.divider()

st.header(
    "Ask a Question"
)


query = st.text_input(
    "Question:",
    placeholder="Example: What happened to Tata Motors?"
)


# ============================================================
# RETRIEVAL
# ============================================================

if query:

    if not os.path.exists(
        file_path
    ):

        st.warning(
            "Please process the URLs first."
        )

        st.stop()


    # ========================================================
    # LOAD FAISS + DOCUMENTS
    # ========================================================

    with open(
        file_path,
        "rb"
    ) as f:

        stored_data = pickle.load(f)


    index = stored_data["index"]

    chunks = stored_data["docs"]


    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    query_vector = encoder.encode(

        [query],

        convert_to_numpy=True
    )


    # ========================================================
    # FAISS SEARCH
    # ========================================================

    D, I = index.search(
        query_vector,
        3
    )


    # ========================================================
    # RETRIEVE TOP 3 CHUNKS
    # ========================================================

    results = [

        chunks[i]

        for i in I[0]

        if i != -1
    ]


    # ========================================================
    # SHOW RETRIEVED CHUNKS
    # ========================================================

    with st.expander(
        "View retrieved chunks"
    ):

        for i, result in zip(
            I[0],
            results
        ):

            st.write(
                f"Chunk index: {i}"
            )

            st.write(
                result.page_content
            )

            st.write("---")


    # ========================================================
    # CREATE CONTEXT
    # ========================================================

    context = "\n\n".join(

        result.page_content

        for result in results
    )


    # ========================================================
    # SEND CONTEXT + QUESTION TO GEMINI
    # ========================================================

    with st.spinner(
        "Generating answer..."
    ):

        response = chain.invoke({

            "context": context,

            "question": query

        })


    # ========================================================
    # EXTRACT ANSWER
    # ========================================================

    content = response.content


    if isinstance(
        content,
        list
    ):

        answer = "\n".join(

            block.get(
                "text",
                ""
            )

            for block in content

            if isinstance(
                block,
                dict
            )

            and block.get(
                "type"
            ) == "text"
        )

    else:

        answer = content


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    st.header(
        "Answer"
    )

    st.write(
        answer
    )


    # ========================================================
    # DISPLAY SOURCES
    # ========================================================

    st.subheader(
        "Sources"
    )


    sources = set()


    for result in results:

        source = result.metadata.get(
            "source",
            "Unknown source"
        )

        sources.add(
            source
        )


    for source in sources:

        st.write(
            source
        )
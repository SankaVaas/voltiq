"""
rag/pipeline.py — RAG pipeline for grid incident Q&A.

Flow:
  1. Load incident reports (PDF / text / markdown)
  2. Chunk documents with LangChain's RecursiveCharacterTextSplitter
  3. Embed with fastembed (CPU-friendly, no API key required)
  4. Store in Qdrant vector database
  5. Retrieve top-k chunks at query time
  6. Pass context + question to LLM for final answer
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


# ── Embeddings (CPU-friendly via fastembed) ───────────────────────────────────

class FastEmbedWrapper:
    """Thin wrapper so fastembed works with LangChain interfaces."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name=model_name)
        self.dimension = 384  # bge-small output dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [emb.tolist() for emb in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return list(self._model.embed([text]))[0].tolist()


# ── Document loading ──────────────────────────────────────────────────────────

def load_documents(source_dir: str | Path) -> list[Document]:
    """Load .txt and .md files from a directory."""
    source_dir = Path(source_dir)
    docs: list[Document] = []

    txt_loader = DirectoryLoader(
        str(source_dir), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    docs.extend(txt_loader.load())

    md_loader = DirectoryLoader(
        str(source_dir), glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, show_progress=True
    )
    docs.extend(md_loader.load())

    logger.info("Loaded documents", count=len(docs), source=str(source_dir))
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Chunked documents", original=len(docs), chunks=len(chunks))
    return chunks


# ── Qdrant vector store ───────────────────────────────────────────────────────

class VoltiqVectorStore:
    """Manages Qdrant collections for grid incident data."""

    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or settings.qdrant_collection_incidents
        self.embedder = FastEmbedWrapper(settings.embedding_model)
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection", name=self.collection_name)

    def ingest(self, chunks: list[Document]) -> int:
        """Embed chunks and upsert into Qdrant. Returns number ingested."""
        from qdrant_client.models import PointStruct
        import uuid

        texts = [c.page_content for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": text,
                    "source": chunk.metadata.get("source", ""),
                    "chunk_index": i,
                },
            )
            for i, (text, vec, chunk) in enumerate(zip(texts, vectors, chunks))
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Ingested chunks into Qdrant", count=len(points), collection=self.collection_name)
        return len(points)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query."""
        query_vector = self.embedder.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": r.payload["text"],
                "source": r.payload.get("source", ""),
                "score": r.score,
            }
            for r in results
        ]


# ── RAG chain ─────────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are Voltiq's grid operations assistant.
Answer the operator's question using ONLY the provided context from incident reports and grid data.
Be precise and technical. If the context does not contain enough information, say so explicitly.
Never fabricate grid statistics or incident details.

Context:
{context}
"""),
    ("human", "{question}"),
])


class GridRAGChain:
    """
    End-to-end RAG chain: retrieve relevant documents → generate grounded answer.

    Usage:
        chain = GridRAGChain()
        answer = chain.invoke("What caused the frequency deviation on 2023-07-14?")
    """

    def __init__(self, collection_name: str | None = None, top_k: int = 5):
        self.vector_store = VoltiqVectorStore(collection_name)
        self.top_k = top_k
        self.llm = ChatAnthropic(
            model=settings.llm_model,
            anthropic_api_key=settings.anthropic_api_key,
            max_tokens=settings.llm_max_tokens,
        )
        self._chain = self._build_chain()

    def _build_chain(self):
        def retrieve(query: str) -> str:
            results = self.vector_store.search(query, top_k=self.top_k)
            return "\n\n---\n\n".join(
                f"[Source: {r['source']}]\n{r['text']}" for r in results
            )

        return (
            {"context": retrieve, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, question: str) -> str:
        logger.info("RAG query", question=question[:100])
        return self._chain.invoke(question)

    def invoke_with_sources(self, question: str) -> dict[str, Any]:
        """Return both the answer and the retrieved source chunks."""
        sources = self.vector_store.search(question, top_k=self.top_k)
        context = "\n\n---\n\n".join(
            f"[Source: {r['source']}]\n{r['text']}" for r in sources
        )
        prompt = RAG_PROMPT.format_messages(context=context, question=question)
        answer = (self.llm | StrOutputParser()).invoke(prompt)
        return {"answer": answer, "sources": sources}


# ── Ingestion helper ──────────────────────────────────────────────────────────

def ingest_incident_reports(source_dir: str | Path) -> int:
    """One-shot: load → chunk → embed → store. Returns total chunks ingested."""
    docs = load_documents(source_dir)
    if not docs:
        logger.warning("No documents found", source=str(source_dir))
        return 0
    chunks = chunk_documents(docs)
    store = VoltiqVectorStore()
    return store.ingest(chunks)

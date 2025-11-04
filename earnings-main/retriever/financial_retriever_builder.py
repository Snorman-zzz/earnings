"""
Financial Retriever Builder using Hybrid BM25 + Vector Search.
Optimized for financial document retrieval with semantic and keyword matching.
"""

import logging
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from config.settings import settings

logger = logging.getLogger(__name__)


class FinancialRetrieverBuilder:
    """
    Build hybrid retrieval system combining:
    - BM25 (keyword/sparse retrieval) - good for exact financial terms
    - Vector search (semantic/dense retrieval) - good for conceptual queries
    """

    def __init__(self):
        """Initialize retriever builder with OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        logger.info(f"OpenAI embeddings initialized: {settings.EMBEDDING_MODEL}")

    def build_hybrid_retriever(
        self, documents: List[Document], weights: List[float] = None
    ) -> EnsembleRetriever:
        """
        Build hybrid retriever combining BM25 and vector search.

        Args:
            documents: List of document chunks
            weights: [bm25_weight, vector_weight]. Defaults to [0.4, 0.6]

        Returns:
            EnsembleRetriever combining both methods
        """
        if weights is None:
            weights = [0.4, 0.6]  # Slightly favor vector search for semantic understanding

        logger.info("=" * 80)
        logger.info(f"Building hybrid retriever from {len(documents)} documents")
        logger.info(f"Weights: BM25={weights[0]}, Vector={weights[1]}")

        # Build BM25 retriever (keyword-based)
        logger.info("Building BM25 retriever...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = settings.RETRIEVAL_K  # Number of docs to retrieve
        logger.info(f"BM25 retriever configured (k={settings.RETRIEVAL_K})")

        # Build Vector retriever (semantic-based)
        logger.info("Building vector store with Chroma...")
        vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.embeddings
        )
        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.RETRIEVAL_K}
        )
        logger.info(f"Vector retriever configured (k={settings.RETRIEVAL_K})")

        # Combine into ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], weights=weights
        )

        logger.info("Hybrid retriever built successfully")
        logger.info("=" * 80)

        return ensemble_retriever

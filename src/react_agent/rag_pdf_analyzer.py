import os
import logging
from typing import Dict, Any, Optional, List
import time
import warnings
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    import Stemmer
    import tabula
    import pandas as pd
    from llama_index.core import Document
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import AutoMergingRetriever
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
    from .rag_config import rag_config
    RAG_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RAG dependencies not available: {e}")
    RAG_DEPENDENCIES_AVAILABLE = False

# Suppress PDF warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")

logger = logging.getLogger(__name__)


class RAGPDFAnalyzer:
    """RAG-based PDF analysis for earnings documents."""

    def __init__(self):
        """Initialize the RAG PDF analyzer with configured models."""
        if not RAG_DEPENDENCIES_AVAILABLE:
            raise ImportError("RAG dependencies are not installed. Please run: pip install -r requirements.txt")
            
        self.config = rag_config
        self.llm = self.config.llm
        self.judge_llm = self.config.judge_llm
        self.embed_model = self.config.embed_model
        
        # Initialize evaluators
        self.faithfulness_evaluator = FaithfulnessEvaluator(llm=self.judge_llm)
        self.relevancy_evaluator = RelevancyEvaluator(llm=self.judge_llm)
        
        logger.info("RAG PDF Analyzer initialized")

    def _extract_pdf_content(self, pdf_bytes: bytes, filename: str) -> List[Document]:
        """Extract content from PDF bytes including text and tables."""
        try:
            # Save bytes to temporary file for processing
            temp_path = f"/tmp/{filename}_{int(time.time())}.pdf"
            with open(temp_path, 'wb') as f:
                f.write(pdf_bytes)
            
            # Extract text content
            documents = SimpleDirectoryReader(input_files=[temp_path]).load_data()
            logger.info(f"Extracted {len(documents)} text documents from {filename}")
            
            # Extract tables using tabula
            try:
                tables = tabula.read_pdf(temp_path, pages="all")
                if tables:
                    table_docs = [df.to_markdown(index=False) for df in tables]
                    all_tables_text = "\n\n".join(table_docs)
                    table_document = Document(text=all_tables_text, metadata={"source": f"{filename}_tables"})
                    documents.append(table_document)
                    logger.info(f"Extracted {len(tables)} tables from {filename}")
            except Exception as e:
                logger.warning(f"Failed to extract tables from {filename}: {str(e)}")
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting content from {filename}: {str(e)}")
            return []

    def _create_rag_pipeline(self, documents: List[Document]) -> Dict[str, Any]:
        """Create RAG pipeline with multiple retrievers."""
        try:
            # Create ingestion pipeline
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(
                        chunk_size=self.config.chunk_size, 
                        chunk_overlap=self.config.chunk_overlap
                    ),
                    self.embed_model
                ]
            )
            
            # Process documents
            nodes = pipeline.run(documents=documents)
            logger.info(f"Created {len(nodes)} nodes from documents")
            
            # Create vector index
            index = VectorStoreIndex(nodes)
            
            # Create multiple retrievers
            base_retriever = index.as_retriever(similarity_top_k=self.config.similarity_top_k)
            
            # Auto-merging retriever
            auto_base_retriever = index.as_retriever(similarity_top_k=self.config.similarity_top_k)
            auto_merging_retriever = AutoMergingRetriever(auto_base_retriever, index.storage_context)
            
            # BM25 retriever
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=self.config.bm25_top_k,
                stemmer=Stemmer.Stemmer("english"),
                language="english"
            )
            
            # Create query engines
            base_query_engine = RetrieverQueryEngine.from_args(base_retriever)
            auto_query_engine = RetrieverQueryEngine.from_args(auto_merging_retriever)
            bm25_query_engine = RetrieverQueryEngine.from_args(bm25_retriever)
            
            return {
                "nodes": nodes,
                "index": index,
                "retrievers": {
                    "base": base_retriever,
                    "auto_merging": auto_merging_retriever,
                    "bm25": bm25_retriever
                },
                "query_engines": {
                    "base": base_query_engine,
                    "auto_merging": auto_query_engine,
                    "bm25": bm25_query_engine
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating RAG pipeline: {str(e)}")
            raise

    def _select_best_response(self, responses: Dict[str, Any], query: str) -> str:
        """Select the best response from multiple retrievers based on evaluation metrics."""
        try:
            best_response = None
            best_score = 0
            
            for retriever_name, response in responses.items():
                try:
                    # Evaluate faithfulness and relevancy
                    faithfulness = self.faithfulness_evaluator.evaluate_response(response=response)
                    relevancy = self.relevancy_evaluator.evaluate_response(query=query, response=response)
                    
                    # Calculate combined score
                    combined_score = (faithfulness.score + relevancy.score) / 2
                    
                    logger.info(f"{retriever_name} - Faithfulness: {faithfulness.score:.3f}, "
                              f"Relevancy: {relevancy.score:.3f}, Combined: {combined_score:.3f}")
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_response = response.response
                        
                except Exception as eval_error:
                    logger.warning(f"Evaluation failed for {retriever_name}: {str(eval_error)}")
                    # Fallback to using the response without evaluation
                    if best_response is None:
                        best_response = response.response
            
            return best_response or "Analysis could not be completed due to evaluation errors."
            
        except Exception as e:
            logger.error(f"Error in response selection: {str(e)}")
            # Return the first available response as fallback
            return next(iter(responses.values())).response

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def analyze_earnings_documents(
            self,
            press_release: bytes,
            presentation: bytes,
            ticker: str,
            company_name: str,
            market_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze earnings documents using RAG pipeline.

        Args:
            press_release: Press release PDF content as bytes
            presentation: Presentation PDF content as bytes
            ticker: Stock ticker symbol
            company_name: Company name
            market_data: Dictionary with market data (eps, revenue estimates, price)

        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        logger.info(f"Starting RAG analysis for {company_name} ({ticker})")

        try:
            # Extract content from both PDFs
            logger.info("Extracting content from PDFs...")
            press_release_docs = self._extract_pdf_content(press_release, "press_release")
            presentation_docs = self._extract_pdf_content(presentation, "presentation")
            
            if not press_release_docs and not presentation_docs:
                return {"error": "Failed to extract content from PDF files"}
            
            # Combine all documents
            all_documents = press_release_docs + presentation_docs
            logger.info(f"Combined {len(all_documents)} documents for analysis")
            
            # Create RAG pipeline
            logger.info("Creating RAG pipeline...")
            rag_pipeline = self._create_rag_pipeline(all_documents)
            
            # Get financial analysis prompt
            prompt = self.config.get_financial_analysis_prompt(company_name, ticker, market_data)
            
            # Query all retrievers
            logger.info("Querying RAG pipeline...")
            responses = {}
            for name, query_engine in rag_pipeline["query_engines"].items():
                try:
                    response = query_engine.query(prompt)
                    responses[name] = response
                    logger.info(f"Got response from {name} retriever")
                except Exception as e:
                    logger.warning(f"Error querying {name} retriever: {str(e)}")
            
            if not responses:
                return {"error": "No responses generated from RAG pipeline"}
            
            # Select best response
            logger.info("Evaluating and selecting best response...")
            analysis_text = self._select_best_response(responses, prompt)
            
            logger.info(f"RAG analysis completed in {time.time() - start_time:.2f} seconds")
            
            # Return result in same format as Claude API
            result = {
                "full_analysis": analysis_text,
                "retrieval_info": {
                    "total_nodes": len(rag_pipeline["nodes"]),
                    "retrievers_used": list(responses.keys()),
                    "processing_time": time.time() - start_time
                }
            }
            
            return result

        except Exception as e:
            logger.error(f"Error in RAG analysis: {str(e)}")
            return {"error": f"Error in RAG analysis: {str(e)}"}

    def get_retrieval_stats(self, query: str, rag_pipeline: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed retrieval statistics for analysis."""
        stats = {}
        
        try:
            for name, retriever in rag_pipeline["retrievers"].items():
                retrieved_nodes = retriever.retrieve(query)
                stats[name] = {
                    "nodes_retrieved": len(retrieved_nodes),
                    "avg_score": sum(node.score for node in retrieved_nodes if hasattr(node, 'score')) / len(retrieved_nodes) if retrieved_nodes else 0,
                    "sources": [node.metadata.get("source", "unknown") for node in retrieved_nodes]
                }
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {str(e)}")
            stats = {"error": str(e)}
        
        return stats
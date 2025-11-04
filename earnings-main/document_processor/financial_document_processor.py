"""
Financial Document Processor using Docling for PDF to Markdown conversion.
Handles earnings reports (press releases and presentations) with table preservation.
"""

import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple
import pickle

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from config.constants import DOC_TYPES

logger = logging.getLogger(__name__)


class FinancialDocumentProcessor:
    """
    Process financial PDF documents (earnings reports) using Docling.
    Converts PDF → Markdown → Chunks with financial table preservation.
    """

    def __init__(self):
        """Initialize document processor with Docling converter and text splitter."""
        # Initialize Docling converter with table OCR enabled
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True  # Enable table structure recognition
        pipeline_options.do_ocr = True  # Enable OCR for scanned documents

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        # Use larger chunk size for financial documents to avoid splitting tables
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
            length_function=len,
        )

        self.cache_dir = settings.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("FinancialDocumentProcessor initialized with Docling")

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file for caching."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_cache_path(self, file_hash: str) -> Path:
        """Get cache file path for a given file hash."""
        return self.cache_dir / f"{file_hash}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid (within expiry period)."""
        if not cache_path.exists():
            return False

        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(days=settings.CACHE_EXPIRY_DAYS)

    def _load_from_cache(self, file_hash: str) -> List[Document]:
        """Load processed chunks from cache."""
        cache_path = self._get_cache_path(file_hash)
        if self._is_cache_valid(cache_path):
            logger.info(f"Loading from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, file_hash: str, chunks: List[Document]):
        """Save processed chunks to cache."""
        cache_path = self._get_cache_path(file_hash)
        with open(cache_path, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Saved to cache: {cache_path}")

    def _convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """
        Convert PDF to Markdown using Docling.
        Preserves tables and document structure.
        """
        logger.info(f"Converting PDF to Markdown: {pdf_path}")

        # Convert using Docling
        result = self.converter.convert(pdf_path)

        # Export to Markdown
        markdown_content = result.document.export_to_markdown()

        logger.info(f"Converted {pdf_path} to Markdown ({len(markdown_content)} characters)")
        return markdown_content

    def _create_chunks_with_metadata(
        self, markdown: str, doc_type: str, source_file: str
    ) -> List[Document]:
        """
        Split Markdown into chunks with metadata.

        Args:
            markdown: Markdown content
            doc_type: "press_release" or "presentation"
            source_file: Original PDF filename

        Returns:
            List of Document objects with metadata
        """
        # Split into chunks
        chunks = self.splitter.create_documents(
            texts=[markdown],
            metadatas=[
                {
                    "source": source_file,
                    "doc_type": doc_type,
                    "doc_name": DOC_TYPES.get(doc_type, doc_type),
                }
            ],
        )

        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks

    def process_earnings_pdfs(
        self, press_release_path: str, presentation_path: str
    ) -> Tuple[List[Document], List[Document]]:
        """
        Process both earnings PDFs (press release and presentation).

        Args:
            press_release_path: Path to press release PDF
            presentation_path: Path to earnings presentation PDF

        Returns:
            Tuple of (press_release_chunks, presentation_chunks)
        """
        logger.info("=" * 80)
        logger.info("Processing earnings documents")
        logger.info(f"Press Release: {press_release_path}")
        logger.info(f"Presentation: {presentation_path}")
        logger.info("=" * 80)

        # Process press release
        pr_hash = self._compute_file_hash(press_release_path)
        pr_chunks = self._load_from_cache(pr_hash)

        if pr_chunks is None:
            pr_markdown = self._convert_pdf_to_markdown(press_release_path)
            pr_chunks = self._create_chunks_with_metadata(
                pr_markdown,
                "press_release",
                Path(press_release_path).name,
            )
            self._save_to_cache(pr_hash, pr_chunks)
        else:
            logger.info(f"Loaded press release from cache ({len(pr_chunks)} chunks)")

        # Process presentation
        pres_hash = self._compute_file_hash(presentation_path)
        pres_chunks = self._load_from_cache(pres_hash)

        if pres_chunks is None:
            pres_markdown = self._convert_pdf_to_markdown(presentation_path)
            pres_chunks = self._create_chunks_with_metadata(
                pres_markdown,
                "presentation",
                Path(presentation_path).name,
            )
            self._save_to_cache(pres_hash, pres_chunks)
        else:
            logger.info(f"Loaded presentation from cache ({len(pres_chunks)} chunks)")

        total_chunks = len(pr_chunks) + len(pres_chunks)
        logger.info(f"Total chunks created: {total_chunks}")

        return pr_chunks, pres_chunks

    def process_all_documents(
        self, press_release_path: str, presentation_path: str
    ) -> List[Document]:
        """
        Process both documents and return combined list of chunks.

        Args:
            press_release_path: Path to press release PDF
            presentation_path: Path to earnings presentation PDF

        Returns:
            Combined list of all document chunks
        """
        pr_chunks, pres_chunks = self.process_earnings_pdfs(
            press_release_path, presentation_path
        )

        # Combine all chunks
        all_chunks = pr_chunks + pres_chunks

        logger.info(f"Combined total: {len(all_chunks)} chunks")
        return all_chunks

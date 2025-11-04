"""
Financial Research Agent using GPT-5.
Extracts financial metrics from retrieved document chunks.
"""

import json
import logging
from typing import Dict, List
from openai import OpenAI
from langchain.schema import Document

from config.settings import settings
from config.constants import FINANCIAL_METRICS

logger = logging.getLogger(__name__)


class FinancialResearchAgent:
    """
    Agent responsible for extracting financial metrics from earnings documents.
    Uses GPT-5 with automatic reasoning for complex financial analysis.
    """

    def __init__(self):
        """Initialize research agent with OpenAI GPT-5."""
        logger.info("Initializing FinancialResearchAgent with GPT-5...")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = settings.GPT5_MODEL
        logger.info("GPT-5 client initialized successfully for research")

    def extract_financial_metrics(
        self, question: str, retrieved_docs: List[Document]
    ) -> Dict:
        """
        Extract financial metrics from retrieved document chunks.

        Args:
            question: The specific metric or question to extract
            retrieved_docs: List of relevant document chunks

        Returns:
            Dictionary containing extracted metrics
        """
        logger.info(f"Extracting metrics for: {question}")
        logger.info(f"Using {len(retrieved_docs)} retrieved documents")

        # Combine retrieved documents into context
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

        logger.debug(f"Combined context length: {len(context)} characters")

        # Create structured prompt for financial extraction
        prompt = self._create_extraction_prompt(question, context)

        # Call GPT-5 for extraction
        try:
            logger.info("Calling GPT-5 for financial extraction...")
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                text={"verbosity": "medium"},
                max_output_tokens=settings.RESEARCH_AGENT_MAX_TOKENS,
            )
            logger.info("GPT-5 response received")

            # Extract and parse response
            result_text = response.output_text.strip()
            logger.debug(f"Raw response: {result_text[:200]}...")

            # Parse JSON response
            try:
                result = json.loads(result_text)
                logger.info("Successfully parsed JSON response")
                return result
            except json.JSONDecodeError:
                logger.warning("Response is not valid JSON, returning as text")
                return {"raw_response": result_text, "parsed": False}

        except Exception as e:
            logger.error(f"Error during financial extraction: {e}")
            return {"error": str(e), "question": question}

    def _create_extraction_prompt(self, question: str, context: str) -> str:
        """
        Create structured prompt for financial metric extraction.

        Args:
            question: The metric or question
            context: Combined document context

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a financial analyst extracting precise metrics from earnings reports.

**Task:** Extract financial data to answer the following question:
{question}

**Instructions:**
1. Find exact numerical values from the provided context
2. Identify whether figures are GAAP or non-GAAP
3. Extract units (M for millions, B for billions)
4. Include current quarter and prior year values when available
5. Note the source (press release vs presentation)

**Context from Earnings Documents:**
{context}

**Output Format (JSON):**
{{
    "metric_name": "string (e.g., 'EPS', 'Revenue')",
    "current_quarter": {{
        "value": float,
        "unit": "M" | "B",
        "gaap_type": "GAAP" | "Non-GAAP" | "Adjusted",
        "quarter": "string (e.g., 'Q4 2024')"
    }},
    "prior_year": {{
        "value": float,
        "unit": "M" | "B",
        "gaap_type": "GAAP" | "Non-GAAP" | "Adjusted",
        "quarter": "string (e.g., 'Q4 2023')"
    }},
    "source_doc": "press_release" | "presentation" | "both",
    "confidence": "high" | "medium" | "low",
    "notes": "any additional context or caveats"
}}

**Respond ONLY with valid JSON. Do not include any other text.**
"""
        return prompt

    def extract_all_metrics(self, retriever, ticker: str, company: str) -> Dict:
        """
        Extract all standard financial metrics using multiple targeted queries.

        Args:
            retriever: The retriever to query documents
            ticker: Stock ticker symbol
            company: Company name

        Returns:
            Dictionary containing all extracted metrics
        """
        logger.info("=" * 80)
        logger.info(f"Extracting all metrics for {company} ({ticker})")
        logger.info("=" * 80)

        extracted_metrics = {
            "ticker": ticker,
            "company": company,
            "metrics": {},
        }

        # Define queries for each key metric
        metric_queries = {
            "eps": "What is the reported EPS (Earnings Per Share) for this quarter? Include both GAAP and non-GAAP values.",
            "revenue": "What is the total revenue for this quarter? Include year-over-year comparison.",
            "operating_income": "What is the operating income for this quarter?",
            "net_income": "What is the net income for this quarter?",
            "gross_margin": "What is the gross margin percentage for this quarter?",
            "guidance": "What is the company's forward guidance for next quarter or next year?",
        }

        # Extract each metric
        for metric_key, query in metric_queries.items():
            logger.info(f"Extracting: {metric_key}")

            # Retrieve relevant documents
            docs = retriever.invoke(query)
            logger.debug(f"Retrieved {len(docs)} documents for {metric_key}")

            # Extract metric
            result = self.extract_financial_metrics(query, docs)

            extracted_metrics["metrics"][metric_key] = result

        logger.info(f"Extraction complete: {len(extracted_metrics['metrics'])} metrics extracted")
        logger.info("=" * 80)

        return extracted_metrics

"""
Financial Verification Agent using GPT-5.
Verifies extracted financial metrics against source documents for accuracy.
"""

import json
import logging
from typing import Dict, List
from openai import OpenAI
from langchain.schema import Document

from config.settings import settings

logger = logging.getLogger(__name__)


class FinancialVerificationAgent:
    """
    Agent responsible for verifying financial data extraction accuracy.
    Critical for preventing numerical errors in financial analysis.
    """

    def __init__(self):
        """Initialize verification agent with OpenAI GPT-5."""
        logger.info("Initializing FinancialVerificationAgent with GPT-5...")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = settings.GPT5_MODEL
        logger.info("GPT-5 client initialized successfully for verification")

    def verify_metrics(
        self, extracted_metrics: Dict, retriever
    ) -> Dict:
        """
        Verify extracted financial metrics against source documents.

        Args:
            extracted_metrics: Dictionary of extracted metrics from research agent
            retriever: Retriever to fetch original source documents

        Returns:
            Verification report with accuracy assessment
        """
        logger.info("=" * 80)
        logger.info("Starting financial metrics verification")
        logger.info("=" * 80)

        verification_results = {
            "overall_verified": False,
            "metrics_verified": {},
            "discrepancies": [],
            "corrections": {},
        }

        # Verify each extracted metric
        for metric_key, metric_data in extracted_metrics.get("metrics", {}).items():
            logger.info(f"Verifying: {metric_key}")

            # Skip if extraction failed
            if "error" in metric_data:
                verification_results["metrics_verified"][metric_key] = {
                    "verified": False,
                    "reason": "extraction_failed",
                }
                continue

            # Create verification query
            if isinstance(metric_data, dict) and "metric_name" in metric_data:
                metric_name = metric_data["metric_name"]
                query = f"Find the exact {metric_name} value in the earnings report"

                # Retrieve source documents
                docs = retriever.invoke(query)

                # Verify against sources
                verification = self._verify_single_metric(metric_data, docs)
                verification_results["metrics_verified"][metric_key] = verification

                if not verification["verified"]:
                    verification_results["discrepancies"].append({
                        "metric": metric_key,
                        "issue": verification.get("issue", "Unknown"),
                    })

                if "correction" in verification:
                    verification_results["corrections"][metric_key] = verification["correction"]

        # Determine overall verification status
        verified_count = sum(
            1
            for v in verification_results["metrics_verified"].values()
            if v.get("verified", False)
        )
        total_count = len(verification_results["metrics_verified"])

        verification_results["overall_verified"] = (
            verified_count == total_count and total_count > 0
        )
        verification_results["verification_rate"] = (
            f"{verified_count}/{total_count}" if total_count > 0 else "0/0"
        )

        logger.info(f"Verification complete: {verification_results['verification_rate']}")
        logger.info("=" * 80)

        return verification_results

    def _verify_single_metric(
        self, metric_data: Dict, source_docs: List[Document]
    ) -> Dict:
        """
        Verify a single financial metric against source documents.

        Args:
            metric_data: Extracted metric data
            source_docs: Retrieved source documents

        Returns:
            Verification result dictionary
        """
        # Combine source documents
        context = "\n\n---\n\n".join([doc.page_content for doc in source_docs])

        # Create verification prompt
        prompt = self._create_verification_prompt(metric_data, context)

        try:
            logger.debug("Calling GPT-5 for verification...")
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                text={"verbosity": "medium"},
                max_output_tokens=settings.VERIFICATION_AGENT_MAX_TOKENS,
            )

            # Parse response
            result_text = response.output_text.strip()

            try:
                verification_result = json.loads(result_text)
                logger.debug(f"Verification result: {verification_result}")
                return verification_result
            except json.JSONDecodeError:
                logger.warning("Verification response is not valid JSON")
                return {
                    "verified": False,
                    "issue": "invalid_response_format",
                    "raw_response": result_text,
                }

        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return {"verified": False, "issue": "verification_error", "error": str(e)}

    def _create_verification_prompt(self, metric_data: Dict, context: str) -> str:
        """
        Create structured prompt for metric verification.

        Args:
            metric_data: The extracted metric to verify
            context: Source document context

        Returns:
            Formatted verification prompt
        """
        prompt = f"""You are a financial data verification specialist.

**Task:** Verify the accuracy of extracted financial data against source documents.

**Extracted Data to Verify:**
{json.dumps(metric_data, indent=2)}

**Source Documents:**
{context}

**Verification Checklist:**
1. Does the numerical value match exactly?
2. Is the unit (M/B) correct?
3. Is the GAAP classification accurate?
4. Is the quarter/period correct?
5. Are there any discrepancies?

**Output Format (JSON):**
{{
    "verified": true | false,
    "numerical_match": true | false,
    "unit_match": true | false,
    "gaap_match": true | false,
    "period_match": true | false,
    "confidence": "high" | "medium" | "low",
    "issue": "string describing any problems found" | null,
    "correction": {{
        "field": "value",
        "extracted": "what was extracted",
        "correct": "what it should be"
    }} | null,
    "notes": "additional verification details"
}}

**Respond ONLY with valid JSON. Do not include any other text.**
"""
        return prompt

    def format_verification_report(self, verification_results: Dict) -> str:
        """
        Format verification results into a human-readable report.

        Args:
            verification_results: Verification results dictionary

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FINANCIAL DATA VERIFICATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall status
        status = "✅ VERIFIED" if verification_results["overall_verified"] else "⚠️ ISSUES FOUND"
        report_lines.append(f"Overall Status: {status}")
        report_lines.append(f"Verification Rate: {verification_results['verification_rate']}")
        report_lines.append("")

        # Individual metrics
        report_lines.append("Metrics Verification:")
        report_lines.append("-" * 80)
        for metric_key, verification in verification_results["metrics_verified"].items():
            verified = verification.get("verified", False)
            icon = "✅" if verified else "❌"
            report_lines.append(f"{icon} {metric_key.upper()}: {'Verified' if verified else 'Failed'}")

            if not verified and "issue" in verification:
                report_lines.append(f"   Issue: {verification['issue']}")

        report_lines.append("")

        # Discrepancies
        if verification_results["discrepancies"]:
            report_lines.append("Discrepancies Found:")
            report_lines.append("-" * 80)
            for disc in verification_results["discrepancies"]:
                report_lines.append(f"• {disc['metric']}: {disc['issue']}")
            report_lines.append("")

        # Corrections
        if verification_results["corrections"]:
            report_lines.append("Suggested Corrections:")
            report_lines.append("-" * 80)
            for metric, correction in verification_results["corrections"].items():
                report_lines.append(f"• {metric}:")
                report_lines.append(f"  Extracted: {correction.get('extracted', 'N/A')}")
                report_lines.append(f"  Correct: {correction.get('correct', 'N/A')}")
            report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)

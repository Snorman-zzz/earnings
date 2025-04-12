import base64
import os
import logging
from typing import List, Dict, Any, Optional
import time
import re
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ClaudePDFAnalyzer:
    """Simple PDF analysis using Anthropic Claude API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with your Anthropic API key."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Provide as argument or set ANTHROPIC_API_KEY environment variable.")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info("Claude PDF Analyzer initialized")

    def encode_file(self, file_content: bytes) -> str:
        """Encode file content to base64."""
        import base64
        return base64.b64encode(file_content).decode("utf-8")

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
        Analyze earnings documents using Claude API.

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
        logger.info(f"Starting analysis for {company_name} ({ticker})")

        # Encode both PDFs
        try:
            press_release_b64 = self.encode_file(press_release)
            presentation_b64 = self.encode_file(presentation)
            logger.info("Successfully encoded PDF files")
        except Exception as e:
            logger.error(f"Error encoding PDFs: {str(e)}")
            return {"error": f"Error encoding PDFs: {str(e)}"}

        # Format market data if provided
        market_context = ""
        if market_data:
            market_context = f"""
            Current market data:
            - Expected EPS: {market_data.get('eps', 'N/A')}
            - Expected Revenue: {market_data.get('revenue', 'N/A')}
            - Current Stock Price: {market_data.get('price', 'N/A')}

            Calculate the surprise percentage for EPS and Revenue as: ((Reported - Expected) / Expected) * 100%
            """

        # Create the prompt
        prompt = f"""
        Please analyze these earnings documents for {company_name} ({ticker}).
        The first document is the press release and the second is the presentation.
        {market_context}

        Extract the following information with precise numerical values:
        1. Reported EPS and Revenue for the current quarter
        2. Other key financial metrics (net income, operating income, gross margin, etc.)
        3. Year-over-Year (Y/Y) changes for all metrics
        4. Forward guidance for next quarter/year if available
        5. Any announced stock splits, dividends, or buybacks

        Format your response with EXACTLY these sections:

        1. Earnings Calls Table comparing:
           - Metric | Expected | Reported | Surprise
           - Include rows for EPS and Revenue with precise values and proper units

        2. Financials Table showing:
           - Metric | Current Quarter | Previous Year | Y/Y Change
           - Include rows for Revenue, Net Income, Diluted EPS, Operating Income, etc.

        3. A concise summary of key findings, performance against expectations, and earnings quality

        4. A price prediction based on the earnings data, using this LaTeX format:
           $\\text{{Price Prediction}} = \\text{{CurrentPrice}} \\times (1 + \\text{{AdjustmentFactor}}) = \\text{{Result}}$

        # IMPORTANT UNIT NORMALIZATION INSTRUCTIONS:
        Before calculating any percentage changes or surprises, normalize units first:
        1. For values with different units (like "11.89B" vs "39.33M"), convert both to the same unit first
        2. Convert all values to the same unit (millions or billions) before calculating percentages
        3. If expected revenue is in billions and reported is in millions, convert both to millions before
           calculating the surprise percentage
        4. For numbers with units (B for billions, M for millions), extract the number part and apply the scale:
           - 1B = 1000M (converting billions to millions)
           - 1M = 0.001B (converting millions to billions)

        Ensure proper spacing between numbers and units (like "13.51B" not "13.51billion").
        Use consistent decimal precision (two decimal places) for all numerical values.
        """

        try:
            # Make the API request with the correct document format
            logger.info("Sending request to Claude API")
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                temperature=0,
                system="You are a financial analyst specialized in earnings report analysis. Be extremely precise with numbers, calculations, and ensure consistent formatting of tables.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": press_release_b64
                                }
                            },
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": presentation_b64
                                }
                            }
                        ]
                    }
                ]
            )

            analysis_text = response.content[0].text
            logger.info(f"Claude analysis completed in {time.time() - start_time:.2f} seconds")

            # Extract tables and other components
            result = self._parse_analysis(analysis_text)
            result["full_analysis"] = analysis_text

            return result

        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            return {"error": f"Error calling Anthropic API: {str(e)}"}

    def _parse_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the analysis text to extract structured data."""
        result = {
            "earnings_table": None,
            "financials_table": None,
            "summary": None,
            "price_prediction": None,
            "latex_formula": None
        }

        # Extract tables (same as your existing code)
        tables = {}
        pattern = r'### ([^\n]+)\s*\n\s*\|([^\n]+)\|\s*\n\s*\|[-:\s\|]+\|\s*\n((?:\|[^\n]+\|\s*\n)*)'

        for match in re.finditer(pattern, analysis_text):
            header = match.group(1).strip()
            table_header = match.group(2).strip()
            table_content = match.group(3).strip()

            # Format the full table
            headers = [h.strip() for h in table_header.split('|') if h.strip()]
            separator = '|' + '|'.join(['-' * (len(h) + 2) for h in headers]) + '|'
            full_table = f"| {' | '.join(headers)} |\n{separator}\n{table_content}"
            tables[header] = full_table

        # Store tables in result
        if "Earnings Calls" in tables:
            result["earnings_table"] = tables["Earnings Calls"]
        if "Financials" in tables:
            result["financials_table"] = tables["Financials"]

        # Extract summary (text outside of tables and formulas)
        summary = re.sub(r'### [^\n]+\s*\n\s*\|[^\n]+\|\s*\n\s*\|[-:\s\|]+\|\s*\n((?:\|[^\n]+\|\s*\n)*)', '',
                         analysis_text)
        summary = re.sub(r'\$\$[^$]+\$\$|\$[^$]+\$', '', summary)  # Remove LaTeX
        summary = re.sub(r'###[^\n]*\n', '', summary)  # Remove headers
        summary = re.sub(r'\n{3,}', '\n\n', summary)  # Normalize spacing
        result["summary"] = summary.strip()

        # Extract LaTeX formula for price prediction
        latex_pattern = r'\$([^$]+)\$'
        latex_matches = re.findall(latex_pattern, analysis_text)
        if latex_matches:
            formula = latex_matches[0]
            result["latex_formula"] = formula

            # Try to extract the price prediction result
            result_match = re.search(r'= (\d+\.\d+)$', formula)
            if result_match:
                result["price_prediction"] = float(result_match.group(1))

        return result